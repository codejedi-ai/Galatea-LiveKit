"""
ElevenLabs TTS plugin for LiveKit agents.
Wraps the livekit-plugins-elevenlabs package.
"""
import logging
import os
from typing import Optional

from livekit.agents import tts
from livekit.plugins import elevenlabs

logger = logging.getLogger(__name__)

class ElevenLabsTTS(elevenlabs.TTS):
    """TTS using ElevenLabs API (cloud)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "eleven_multilingual_v2",
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Prioritize api_key passed in, then ENV
        resolved_api_key = (api_key or os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY") or "").strip().strip('"').strip("'")
        
        # Handle optimize_streaming_latency mapping
        if "optimize_streaming_latency" in kwargs:
            kwargs["streaming_latency"] = kwargs.pop("optimize_streaming_latency")
            
        super().__init__(
            model=model,
            voice_id=voice_id,
            api_key=resolved_api_key,
            **kwargs
        )
        logger.info(f"Initialized ElevenLabsTTS with model={model}, voice_id={voice_id}")
