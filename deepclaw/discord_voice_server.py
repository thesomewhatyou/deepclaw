"""Discord voice channel integration for deepclaw.

Bridges Discord voice channels to the Deepgram Voice Agent API,
using OpenClaw as the LLM backend.

Architecture:

  Discord Voice Channel  (Opus 48 kHz stereo)
        │ decoded to PCM by discord.py, converted to mono
        ▼
  deepclaw Discord Bot
        │  linear16 48 kHz mono
        ▼
  Deepgram Voice Agent API  ←──/v1/chat/completions──→  OpenClaw Gateway
        │  linear16 48 kHz mono
        ▼
  deepclaw Discord Bot
        │  mono→stereo, encoded to Opus by discord.py
        ▼
  Discord Voice Channel

The bot responds to two slash commands:
  /join  — join the caller's voice channel and start a Deepgram session
  /leave — leave and clean up
"""

import array
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Optional

import discord
from discord.ext import commands
from discord.sinks import Sink as DiscordSink
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import httpx
import uvicorn
import websockets

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENCLAW_GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
OPENCLAW_GATEWAY_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
# Public-facing URL of *this* server so Deepgram can reach /v1/chat/completions.
# Typically the output of `ngrok http <PORT>`.
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENCLAW_VOICE_MODEL = os.getenv(
    "OPENCLAW_VOICE_MODEL", "anthropic/claude-haiku-4-5-20251001"
)

DEEPGRAM_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"

# Discord audio parameters (what discord.py gives us after Opus decoding)
DISCORD_SAMPLE_RATE = 48_000   # Hz
DISCORD_CHANNELS = 2           # stereo
DISCORD_SAMPLE_WIDTH = 2       # bytes per sample (16-bit signed)
DISCORD_FRAME_MS = 20          # ms per audio frame
# Bytes per 20 ms stereo frame: 48000 * 2 ch * 2 bytes * 0.020 s = 3840
DISCORD_FRAME_STEREO = (
    DISCORD_SAMPLE_RATE * DISCORD_CHANNELS * DISCORD_SAMPLE_WIDTH * DISCORD_FRAME_MS
) // 1000
# Bytes per 20 ms mono frame (for Deepgram input)
DISCORD_FRAME_MONO = DISCORD_FRAME_STEREO // DISCORD_CHANNELS  # 1920

# ---------------------------------------------------------------------------
# Audio helpers — pure-Python, no audioop (removed in Python 3.13)
# ---------------------------------------------------------------------------

try:
    import audioop as _audioop  # type: ignore[import]

    def pcm_stereo_to_mono(data: bytes) -> bytes:
        """Average left+right channels into a single mono channel."""
        return _audioop.tomono(data, DISCORD_SAMPLE_WIDTH, 0.5, 0.5)

    def pcm_mono_to_stereo(data: bytes) -> bytes:
        """Duplicate the mono channel into left and right."""
        return _audioop.tostereo(data, DISCORD_SAMPLE_WIDTH, 1, 1)

except ImportError:
    # Python 3.13+: audioop was removed; fall back to array module

    def pcm_stereo_to_mono(data: bytes) -> bytes:  # type: ignore[misc]
        """Average left+right channels into a single mono channel."""
        samples: array.array = array.array("h", data)
        if sys.byteorder != "little":
            samples.byteswap()
        mono: array.array = array.array(
            "h",
            [(samples[i] + samples[i + 1]) >> 1 for i in range(0, len(samples), 2)],
        )
        if sys.byteorder != "little":
            mono.byteswap()
        return mono.tobytes()

    def pcm_mono_to_stereo(data: bytes) -> bytes:  # type: ignore[misc]
        """Duplicate the mono channel into left and right."""
        samples: array.array = array.array("h", data)
        if sys.byteorder != "little":
            samples.byteswap()
        stereo: array.array = array.array(
            "h", [val for s in samples for val in (s, s)]
        )
        if sys.byteorder != "little":
            stereo.byteswap()
        return stereo.tobytes()


# ---------------------------------------------------------------------------
# FastAPI — LLM proxy (same pattern as voice_agent_server.py)
# ---------------------------------------------------------------------------

app = FastAPI(title="deepclaw-discord-voice")

# Maps guild_id (str) → OpenClaw session key.
# "_current" is a catch-all used by the proxy when no guild is identified.
_active_sessions: dict[str, str] = {}


def strip_markdown(text: str) -> str:
    """Strip markdown formatting so spoken text sounds natural."""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0001F900-\U0001F9FF]+",
        "",
        text,
    )
    text = re.sub(r"\n+", " ", text)
    return text


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Proxy LLM requests from Deepgram Voice Agent to local OpenClaw.

    Deepgram calls this endpoint; we forward the request to OpenClaw,
    stripping markdown from streamed chunks so TTS output sounds natural.
    """
    logger.info("LLM proxy request received")
    body = await request.json()

    # Route to the dedicated 'voice' OpenClaw agent
    body["model"] = "openclaw/voice"

    stream = body.get("stream", False)
    logger.info(
        "Proxying chat completion — stream=%s, messages=%d",
        stream,
        len(body.get("messages", [])),
    )

    session_key = _active_sessions.get("_current")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENCLAW_GATEWAY_TOKEN}",
    }
    if session_key:
        headers["X-OpenClaw-Session-Key"] = session_key
        logger.info("Using session key: %s", session_key)

    async def stream_response():
        chunk_count = 0
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{OPENCLAW_GATEWAY_URL}/v1/chat/completions",
                json=body,
                headers=headers,
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info("First chunk received from OpenClaw")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    delta["content"] = strip_markdown(
                                        delta["content"]
                                    )
                            yield f"data: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError as exc:
                            logger.warning("Malformed SSE data: %s", exc)
                            yield f"{line}\n\n"
                    elif line.strip():
                        yield f"{line}\n\n"
        logger.info("Stream complete: %d chunks", chunk_count)

    if stream:
        return StreamingResponse(
            stream_response(), media_type="text/event-stream"
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENCLAW_GATEWAY_URL}/v1/chat/completions",
            json=body,
            headers=headers,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type="application/json",
        )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "deepclaw-discord-voice"}


# ---------------------------------------------------------------------------
# Deepgram agent configuration — tuned for Discord (linear16 48 kHz)
# ---------------------------------------------------------------------------

def get_agent_config(public_url: str) -> dict:
    """
    Build the Deepgram Voice Agent Settings message for Discord.

    We use linear16 at 48 kHz for both input and output so no resampling
    is required; discord.py decodes Opus to exactly this format.
    """
    llm_url = f"{public_url}/v1/chat/completions"
    return {
        "type": "Settings",
        "audio": {
            "input": {
                "encoding": "linear16",
                "sample_rate": DISCORD_SAMPLE_RATE,
            },
            "output": {
                "encoding": "linear16",
                "sample_rate": DISCORD_SAMPLE_RATE,
                "container": "none",
            },
        },
        "agent": {
            "language": "en",
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": "nova-2-general",
                },
            },
            "think": {
                "provider": {
                    "type": "open_ai",
                    "model": "gpt-4o-mini",
                },
                "endpoint": {
                    "url": llm_url,
                },
                "prompt": (
                    "You are a helpful voice assistant in a Discord voice channel. "
                    "Keep responses concise and conversational (1-3 sentences). "
                    "Never use markdown, bullet points, numbered lists, or emojis — "
                    "your responses will be spoken aloud."
                ),
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": "aura-2-thalia-en",
                },
            },
            "greeting": "Hello! I'm here. How can I help you?",
        },
    }


# ---------------------------------------------------------------------------
# Discord audio sink — receives decoded PCM from Discord, queues for Deepgram
# ---------------------------------------------------------------------------

class DiscordToDeepgramSink(DiscordSink):
    """
    Custom Discord audio sink that forwards decoded PCM to Deepgram.

    discord.py decodes incoming Opus packets from every speaker and calls
    ``write(data, user)`` from a background voice-receive thread.  We
    convert stereo→mono and enqueue the mono frame for the async send loop.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop

    def write(self, data, user) -> None:  # noqa: ANN001
        """Called per audio packet by discord.py's voice-receive thread."""
        # data may be a RawData namedtuple (.data) or raw bytes depending on
        # the discord.py version.
        pcm: bytes = getattr(data, "data", data)
        if not pcm:
            return
        mono = pcm_stereo_to_mono(pcm)
        # Bridge from the sync receiver thread to the async event loop
        asyncio.run_coroutine_threadsafe(self._queue.put(mono), self._loop)

    def cleanup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Continuous PCM audio source for Discord playback
# ---------------------------------------------------------------------------

class BufferedPCMAudio(discord.AudioSource):
    """
    Continuous PCM audio source backed by a thread-safe rolling buffer.

    Produces 20 ms frames of 48 kHz stereo int16 PCM (3840 bytes/frame).
    Returns silence when the buffer is empty so discord.py's player never
    stops, allowing us to push audio asynchronously.
    """

    FRAME_SIZE = DISCORD_FRAME_STEREO  # 3840 bytes

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def add(self, data: bytes) -> None:
        """Append stereo PCM bytes (called from the async event loop)."""
        with self._lock:
            self._buffer.extend(data)

    def clear(self) -> None:
        """Discard buffered audio — used for barge-in."""
        with self._lock:
            self._buffer.clear()

    def read(self) -> bytes:
        """Called by discord.py's audio-player thread every 20 ms."""
        with self._lock:
            if len(self._buffer) < self.FRAME_SIZE:
                return bytes(self.FRAME_SIZE)  # silence
            frame = bytes(self._buffer[: self.FRAME_SIZE])
            del self._buffer[: self.FRAME_SIZE]
            return frame

    def is_opus(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Voice session — one per Discord guild
# ---------------------------------------------------------------------------

class VoiceSession:
    """Manages the Deepgram ↔ Discord voice bridge for one guild."""

    def __init__(self, vc: discord.VoiceClient, guild_id: int) -> None:
        self.vc = vc
        self.guild_id = guild_id
        self.session_key = f"agent:voice:discord:{guild_id}"

        self.deepgram_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._audio_in: asyncio.Queue[bytes] = asyncio.Queue()
        self._audio_out = BufferedPCMAudio()

        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Connect to Deepgram and start the full audio bridge."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        # Register session key so the LLM proxy can attach it to OpenClaw
        _active_sessions[str(self.guild_id)] = self.session_key
        _active_sessions["_current"] = self.session_key

        # Connect to Deepgram Voice Agent API
        self.deepgram_ws = await websockets.connect(
            DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        )
        logger.info("[Guild %d] Connected to Deepgram Voice Agent API", self.guild_id)

        # Send agent configuration with the public LLM callback URL
        config = get_agent_config(PUBLIC_URL)
        await self.deepgram_ws.send(json.dumps(config))
        logger.info("[Guild %d] Sent agent config", self.guild_id)

        # Attach voice audio sink to start capturing from Discord
        sink = DiscordToDeepgramSink(self._audio_in, self._loop)
        self.vc.start_recording(sink, self._on_recording_done, None)

        # Start the continuous playback source so Deepgram TTS is heard
        self.vc.play(self._audio_out, after=self._on_playback_end)

        # Launch background coroutines
        self._send_task = asyncio.create_task(
            self._send_loop(), name=f"deepclaw-send-{self.guild_id}"
        )
        self._recv_task = asyncio.create_task(
            self._recv_loop(), name=f"deepclaw-recv-{self.guild_id}"
        )
        logger.info("[Guild %d] Voice session active", self.guild_id)

    # ------------------------------------------------------------------
    # Callbacks (called from discord.py threads)
    # ------------------------------------------------------------------

    def _on_recording_done(self, sink, channel=None) -> None:  # noqa: ANN001
        """Called by discord.py when stop_recording() finishes (no-op here)."""

    def _on_playback_end(self, error: Optional[Exception]) -> None:
        if error:
            logger.error("[Guild %d] Playback error: %s", self.guild_id, error)

    # ------------------------------------------------------------------
    # Async audio loops
    # ------------------------------------------------------------------

    async def _send_loop(self) -> None:
        """
        Send 20 ms mono linear16 frames to Deepgram at a steady 50 fps cadence.

        Sends silence between speech packets so Deepgram's VAD has a
        continuous stream and can detect pauses reliably.
        """
        silence = bytes(DISCORD_FRAME_MONO)
        leftover = bytearray()

        while self._running:
            # Drain every packet that arrived since the last iteration
            while True:
                try:
                    chunk = self._audio_in.get_nowait()
                    leftover.extend(chunk)
                except asyncio.QueueEmpty:
                    break

            # Ship exactly one 20 ms frame (real audio or silence)
            if len(leftover) >= DISCORD_FRAME_MONO:
                frame = bytes(leftover[:DISCORD_FRAME_MONO])
                del leftover[:DISCORD_FRAME_MONO]
            else:
                frame = silence

            try:
                await self.deepgram_ws.send(frame)
            except Exception as exc:
                logger.error("[Guild %d] Deepgram send error: %s", self.guild_id, exc)
                break

            await asyncio.sleep(0.02)  # 20 ms cadence

    async def _recv_loop(self) -> None:
        """
        Receive messages from Deepgram and handle them.

        Binary messages are TTS audio (linear16 48 kHz mono) — we convert
        them to stereo and push into the playback buffer.
        Text messages are JSON events (Welcome, ConversationText, etc.).
        """
        while self._running:
            try:
                message = await self.deepgram_ws.recv()

                if isinstance(message, bytes):
                    # TTS audio: mono linear16 48 kHz → convert to stereo
                    stereo = pcm_mono_to_stereo(message)
                    self._audio_out.add(stereo)
                else:
                    event = json.loads(message)
                    etype = event.get("type", "")

                    if etype == "Welcome":
                        logger.info("[Guild %d] Deepgram: Welcome", self.guild_id)
                    elif etype == "SettingsApplied":
                        logger.info("[Guild %d] Deepgram: SettingsApplied", self.guild_id)
                    elif etype == "UserStartedSpeaking":
                        logger.debug("[Guild %d] Barge-in detected", self.guild_id)
                        # Clear queued TTS audio so the bot stops mid-sentence
                        self._audio_out.clear()
                    elif etype == "AgentStartedSpeaking":
                        logger.debug("[Guild %d] Agent speaking", self.guild_id)
                    elif etype == "ConversationText":
                        role = event.get("role", "")
                        content = event.get("content", "")
                        logger.info(
                            "[Guild %d] %s: %s",
                            self.guild_id,
                            role.capitalize(),
                            content,
                        )
                    elif etype == "Error":
                        logger.error("[Guild %d] Deepgram error: %s", self.guild_id, event)

            except websockets.exceptions.ConnectionClosed:
                logger.info("[Guild %d] Deepgram connection closed", self.guild_id)
                break
            except Exception as exc:
                logger.error("[Guild %d] Deepgram recv error: %s", self.guild_id, exc)
                break

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Disconnect from Deepgram and leave the Discord voice channel."""
        self._running = False

        for task in (self._send_task, self._recv_task):
            if task:
                task.cancel()

        if self.vc.is_recording():
            self.vc.stop_recording()
        if self.vc.is_playing():
            self.vc.stop()

        if self.deepgram_ws:
            await self.deepgram_ws.close()
        if self.vc.is_connected():
            await self.vc.disconnect()

        _active_sessions.pop("_current", None)
        _active_sessions.pop(str(self.guild_id), None)
        logger.info("[Guild %d] Session stopped and cleaned up", self.guild_id)


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.voice_states = True

bot = commands.Bot(command_prefix="!", intents=intents)

# guild_id → active VoiceSession
_voice_sessions: dict[int, VoiceSession] = {}


@bot.event
async def on_ready() -> None:
    logger.info("Discord bot ready: %s (id=%s)", bot.user, bot.user.id)
    try:
        synced = await bot.tree.sync()
        logger.info("Synced %d application command(s)", len(synced))
    except Exception as exc:
        logger.error("Command sync failed: %s", exc)


@bot.tree.command(
    name="join",
    description="Have deepclaw join your voice channel and start listening",
)
async def cmd_join(interaction: discord.Interaction) -> None:
    if not interaction.guild:
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return

    voice_state = interaction.user.voice  # type: ignore[union-attr]
    if not voice_state or not voice_state.channel:
        await interaction.response.send_message(
            "You must be in a voice channel first.", ephemeral=True
        )
        return

    guild_id = interaction.guild.id
    if guild_id in _voice_sessions:
        await interaction.response.send_message(
            "Already in a voice channel — use `/leave` first.", ephemeral=True
        )
        return

    if not PUBLIC_URL:
        await interaction.response.send_message(
            "⚠️ `PUBLIC_URL` is not configured. "
            "Run `ngrok http {port}` and set `PUBLIC_URL=https://<ngrok-url>` "
            "in your `.env` so Deepgram can reach the LLM proxy.".format(port=PORT),
            ephemeral=True,
        )
        return

    await interaction.response.defer()

    try:
        vc = await voice_state.channel.connect()
        session = VoiceSession(vc, guild_id)
        _voice_sessions[guild_id] = session
        await session.start()
        await interaction.followup.send(
            f"🎙️ Joined **{voice_state.channel.name}**. "
            "Say something to talk to your OpenClaw!"
        )
        logger.info(
            "Joined voice channel '%s' in guild %d", voice_state.channel.name, guild_id
        )
    except Exception as exc:
        logger.error("Failed to join voice channel: %s", exc)
        _voice_sessions.pop(guild_id, None)
        await interaction.followup.send(f"❌ Failed to join: {exc}")


@bot.tree.command(
    name="leave",
    description="Have deepclaw leave the voice channel",
)
async def cmd_leave(interaction: discord.Interaction) -> None:
    if not interaction.guild:
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return

    guild_id = interaction.guild.id
    session = _voice_sessions.pop(guild_id, None)
    if not session:
        await interaction.response.send_message(
            "Not in a voice channel.", ephemeral=True
        )
        return

    await session.stop()
    await interaction.response.send_message("👋 Left the voice channel. Goodbye!")


# ---------------------------------------------------------------------------
# OpenClaw voice agent provisioning (mirrors voice_agent_server.py)
# ---------------------------------------------------------------------------

def ensure_openclaw_voice_agent() -> None:
    """Create the 'voice' OpenClaw agent if it doesn't already exist."""
    openclaw = shutil.which("openclaw")
    if not openclaw:
        logger.warning(
            "openclaw CLI not found on PATH — skipping voice agent provisioning. "
            "Create it manually: openclaw agents add voice --model %s",
            OPENCLAW_VOICE_MODEL,
        )
        return

    try:
        result = subprocess.run(
            [openclaw, "agents", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "voice" in result.stdout.split():
            logger.info("OpenClaw 'voice' agent already exists")
            return
    except Exception as exc:
        logger.warning("Could not list OpenClaw agents: %s", exc)
        return

    logger.info(
        "Creating OpenClaw 'voice' agent with model %s", OPENCLAW_VOICE_MODEL
    )
    try:
        workspace = os.path.join(
            os.path.expanduser("~"), ".openclaw", "workspace-voice"
        )
        subprocess.run(
            [
                openclaw,
                "agents",
                "add",
                "voice",
                "--model",
                OPENCLAW_VOICE_MODEL,
                "--workspace",
                workspace,
                "--non-interactive",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        logger.info("OpenClaw 'voice' agent created successfully")
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Failed to create OpenClaw voice agent: %s\n%s", exc, exc.stderr
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run deepclaw in Discord voice mode."""
    if not DEEPGRAM_API_KEY:
        logger.error(
            "DEEPGRAM_API_KEY is not set. Get one at https://console.deepgram.com/"
        )
        return
    if not OPENCLAW_GATEWAY_TOKEN:
        logger.error("OPENCLAW_GATEWAY_TOKEN is not set.")
        return
    if not DISCORD_BOT_TOKEN:
        logger.error(
            "DISCORD_BOT_TOKEN is not set. "
            "Create a bot at https://discord.com/developers/applications"
        )
        return
    if not PUBLIC_URL:
        logger.warning(
            "PUBLIC_URL is not set — Deepgram LLM callbacks will fail. "
            "Run `ngrok http %d` and set PUBLIC_URL=https://<ngrok-url> in .env.",
            PORT,
        )

    ensure_openclaw_voice_agent()

    logger.info("Starting deepclaw Discord voice server on %s:%d", HOST, PORT)
    logger.info("Public URL for Deepgram callbacks: %s", PUBLIC_URL or "(not set)")

    async def run_all() -> None:
        uvicorn_config = uvicorn.Config(
            app, host=HOST, port=PORT, log_level="info"
        )
        server = uvicorn.Server(uvicorn_config)
        # Run the FastAPI server and the Discord bot on the same event loop
        await asyncio.gather(
            server.serve(),
            bot.start(DISCORD_BOT_TOKEN),
        )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
