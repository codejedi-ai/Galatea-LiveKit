import logging
import os
from typing import Optional, List
from livekit.plugins import elevenlabs
from .base import BaseProvider, Model

logger = logging.getLogger("provider-elevenlabs")

class ElevenLabsTTS(elevenlabs.TTS):
    def __init__(self, *, voice_id: str = "EXAVITQu4vr4xnSDxMaL", model: str = "eleven_multilingual_v2", **kwargs):
        # Use ELEVEN_API_KEY from .env as per docs
        api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        super().__init__(model=model, voice_id=voice_id, api_key=api_key, **kwargs)

class ElevenLabsSTT(elevenlabs.STT):
    def __init__(self, *, model_id: str = "scribe_v2_realtime"):
        # Use ELEVEN_API_KEY from .env as per docs
        api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        super().__init__(model_id=model_id, api_key=api_key)

class ElevenLabsProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("ElevenLabs", api_key or os.getenv("ELEVEN_API_KEY"))
        self.models = [
            Model("eleven_multilingual_v2", "Multilingual V2", ["text"], ["voice"], "elevenlabs", is_instruct=False, base="custom"),
            Model("eleven_turbo_v2_5", "Turbo V2.5", ["text"], ["voice"], "elevenlabs", is_instruct=False, base="custom"),
            Model("eleven_flash_v2_5", "Flash V2.5", ["text"], ["voice"], "elevenlabs", is_instruct=False, base="custom"),
            Model("scribe_v2_realtime", "Scribe V2 Realtime", ["voice"], ["text"], "elevenlabs", is_instruct=False, base="custom"),
            Model("dubbing_v1", "Dubbing V1", ["video", "voice"], ["video", "voice"], "elevenlabs", is_instruct=False, base="custom"),
        ]

    async def validate(self) -> bool:
        return bool(self.api_key)
