"""Run deepclaw voice agent server.

Set VOICE_PROVIDER in your .env to select a transport:
  VOICE_PROVIDER=twilio    (default) — phone calls via Twilio
  VOICE_PROVIDER=telnyx             — phone calls via Telnyx
  VOICE_PROVIDER=discord            — Discord voice channels
"""

import os

_provider = os.getenv("VOICE_PROVIDER", "twilio").lower()

if _provider == "discord":
    from .discord_voice_server import main
else:
    from .voice_agent_server import main  # type: ignore[assignment]

if __name__ == "__main__":
    main()
