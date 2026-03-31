"""
Smallest AI TTS plugin for LiveKit agents.
Uses the Smallest AI Waves API (cloud) for text-to-speech synthesis.
Requires: pip install smallestai
API key: set SMALLEST_API_KEY in .env or pass directly.
Docs: https://github.com/smallest-inc/smallest-python-sdk
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

logger = logging.getLogger(__name__)

# Smallest AI Waves default sample rate (24 kHz)
SMALLEST_TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1


@dataclass
class SmallestTTSOptions:
    model: str
    voice_id: str
    speed: float
    sample_rate: int
    api_key: str


class SmallestTTS(tts.TTS):
    """TTS using Smallest AI Waves API (cloud)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "lightning",
        voice_id: str = "emily",
        speed: float = 1.0,
        sample_rate: int = SMALLEST_TTS_SAMPLE_RATE,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=TTS_CHANNELS,
        )
        resolved_key = (api_key or os.getenv("SMALLEST_API_KEY", "")).strip().strip('"').strip("'")
        if not resolved_key:
            raise ValueError(
                "SMALLEST_API_KEY is not set. "
                "Set it in .env or pass api_key= when using Smallest AI TTS."
            )
        self._opts = SmallestTTSOptions(
            model=model,
            voice_id=voice_id,
            speed=speed,
            sample_rate=sample_rate,
            api_key=resolved_key,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

    @property
    def provider(self) -> str:
        return "smallestai"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SmallestTTSStream":
        return SmallestTTSStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
        )


def _synthesize_sync(
    api_key: str,
    model: str,
    voice_id: str,
    speed: float,
    sample_rate: int,
    text: str,
) -> tuple[bytes, int]:
    """Run Smallest AI TTS synchronously in a thread; returns (pcm_bytes, sample_rate)."""
    from smallestai.waves import WavesClient

    client = WavesClient(
        api_key=api_key,
        model=model,
        sample_rate=sample_rate,
        voice_id=voice_id,
        speed=speed,
        add_wav_header=False,  # raw PCM for streaming
    )
    # synthesize returns raw audio bytes (PCM 16-bit)
    audio_bytes = client.synthesize(text)

    # The SDK returns PCM int16 bytes by default when add_wav_header=False
    if isinstance(audio_bytes, (bytes, bytearray)):
        return bytes(audio_bytes), sample_rate

    raise ValueError("Smallest AI TTS returned unexpected type: %s" % type(audio_bytes))


class SmallestTTSStream(tts.ChunkedStream):
    """ChunkedStream that calls the Smallest AI Waves API."""

    def __init__(
        self,
        *,
        tts_instance: SmallestTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: SmallestTTSOptions,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._opts = opts

    async def _run(self, output_emitter=None) -> None:
        """Synthesize text via the Smallest AI Waves API and emit audio frames.

        Supports both the old-style (_event_ch) and new-style (output_emitter) APIs
        so it works across LiveKit agents SDK versions.
        """
        request_id = utils.shortuuid()
        loop = asyncio.get_event_loop()

        try:
            start_time = time.time()
            pcm_bytes, sample_rate = await loop.run_in_executor(
                None,
                _synthesize_sync,
                self._opts.api_key,
                self._opts.model,
                self._opts.voice_id,
                self._opts.speed,
                self._opts.sample_rate,
                self.input_text,
            )
        except Exception as exc:
            logger.exception("Smallest AI TTS failed: %s", exc)
            raise APIConnectionError() from exc

        # New-style output_emitter API (LiveKit agents >= 1.x)
        if output_emitter is not None:
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=sample_rate,
                num_channels=TTS_CHANNELS,
                mime_type="audio/pcm",
            )
            output_emitter.push(pcm_bytes)
        else:
            # Old-style _event_ch API (LiveKit agents 0.x)
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=sample_rate,
                num_channels=TTS_CHANNELS,
            )
            for frame in audio_bstream.write(pcm_bytes):
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(frame=frame, request_id=request_id)
                )
            for frame in audio_bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(frame=frame, request_id=request_id)
                )

        logger.info(
            "Smallest AI TTS synthesis completed in %.1fms (model=%s voice=%s)",
            (time.time() - start_time) * 1000,
            self._opts.model,
            self._opts.voice_id,
        )
