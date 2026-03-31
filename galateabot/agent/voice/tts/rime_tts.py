"""
Rime TTS plugin for LiveKit agents.
Wraps the livekit-plugins-rime package.
"""
import logging
import os
from typing import Optional

from livekit.agents import tts
from livekit.plugins import rime

logger = logging.getLogger(__name__)

class RimeTTS(rime.TTS):
    """TTS using Rime API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "arcana",
        speaker: str = "celeste",
        speed_alpha: float = 1.5,
        reduce_latency: bool = True,
        max_tokens: int = 3400,
        **kwargs,
    ) -> None:
        # Prioritize api_key passed in, then ENV
        resolved_api_key = (api_key or os.getenv("RIME_API_KEY") or "").strip().strip('"').strip("'")
        
        super().__init__(
            model=model,
            speaker=speaker,
            speed_alpha=speed_alpha,
            reduce_latency=reduce_latency,
            max_tokens=max_tokens,
            api_key=resolved_api_key,
            **kwargs
        )
        logger.info(f"Initialized RimeTTS with model={model}, speaker={speaker}, speed_alpha={speed_alpha}")
