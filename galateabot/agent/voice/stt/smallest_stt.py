"""
Smallest AI STT plugin for LiveKit agents.
Uses the Smallest AI Pulse API (cloud) for speech-to-text transcription.
REST endpoint: POST https://waves-api.smallest.ai/api/v1/pulse/get_text
API key: set SMALLEST_API_KEY in .env or pass directly.
Docs: https://smallest.ai
"""
import asyncio
import io
import logging
import os
import struct
import wave
from typing import Optional

import numpy as np

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit import rtc

logger = logging.getLogger(__name__)

SMALLEST_STT_SAMPLE_RATE = 16000
PULSE_API_URL = "https://waves-api.smallest.ai/api/v1/pulse/get_text"


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Wrap raw PCM int16 bytes in a proper WAV container for the API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _transcribe_sync(api_key: str, language: str, pcm_bytes: bytes, sample_rate: int) -> str:
    """Call Smallest AI Pulse REST API synchronously; returns transcribed text."""
    import httpx

    wav_data = _pcm_to_wav_bytes(pcm_bytes, sample_rate)

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    files = {
        "file": ("audio.wav", wav_data, "audio/wav"),
    }
    data = {
        "language": language,
    }

    with httpx.Client(timeout=httpx.Timeout(connect=15.0, read=60.0, write=10.0, pool=10.0)) as client:
        response = client.post(PULSE_API_URL, headers=headers, files=files, data=data)
        response.raise_for_status()
        result = response.json()

    # The Pulse API returns JSON with a "text" or "transcript" field
    text = result.get("text") or result.get("transcript") or result.get("transcription") or ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    return str(text).strip()


class SmallestSTTSpeechStream(stt.SpeechStream):
    """RecognizeStream that buffers audio, runs Smallest AI Pulse STT on flush."""

    def __init__(
        self,
        *,
        stt_instance: "SmallestSTT",
        conn_options: APIConnectOptions,
        sample_rate: Optional[int] = None,
        language: str = "en",
        api_key: str = "",
    ) -> None:
        super().__init__(
            stt=stt_instance,
            conn_options=conn_options,
            sample_rate=sample_rate if sample_rate is not None else NOT_GIVEN,
        )
        self._language = language
        self._api_key = api_key

    async def _run(self) -> None:
        def is_flush(item):
            return type(item).__name__ == "_FlushSentinel"

        buffer: list[bytes] = []
        sr = 0
        loop = asyncio.get_event_loop()
        try:
            async for item in self._input_ch:
                if is_flush(item):
                    if not buffer or sr <= 0:
                        continue
                    pcm = b"".join(buffer)
                    buffer.clear()
                    request_id = utils.shortuuid()
                    try:
                        text = await loop.run_in_executor(
                            None,
                            _transcribe_sync,
                            self._api_key,
                            self._language,
                            pcm,
                            sr,
                        )
                    except Exception as exc:
                        logger.exception("Smallest AI STT failed: %s", exc)
                        raise APIConnectionError() from exc
                    duration_sec = len(pcm) / (2 * sr) if sr else 0
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=request_id,
                            alternatives=[stt.SpeechData(language=self._language, text=text.strip() or "")],
                            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
                        )
                    )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.RECOGNITION_USAGE,
                            request_id=request_id,
                            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
                        )
                    )
                else:
                    assert isinstance(item, rtc.AudioFrame)
                    buffer.append(bytes(item.data))
                    if sr <= 0:
                        sr = item.sample_rate
        except Exception as exc:
            logger.exception("Smallest AI STT stream error: %s", exc)
            raise


class SmallestSTT(stt.STT):
    """STT using Smallest AI Pulse API (cloud)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self._language = language
        resolved_key = (api_key or os.getenv("SMALLEST_API_KEY", "")).strip().strip('"').strip("'")
        if not resolved_key:
            raise ValueError(
                "SMALLEST_API_KEY is not set. "
                "Set it in .env or pass api_key= when using Smallest AI STT."
            )
        self._api_key = resolved_key

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        from livekit.agents.utils.misc import is_given

        frames = buffer if isinstance(buffer, list) else [buffer]
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=utils.shortuuid(),
                alternatives=[stt.SpeechData(language=self._language, text="")],
                recognition_usage=stt.RecognitionUsage(audio_duration=0.0),
            )
        lang = self._language
        if is_given(language):
            lang = language or "en"
        sr = frames[0].sample_rate
        pcm = b"".join(bytes(f.data) for f in frames)
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None, _transcribe_sync, self._api_key, lang, pcm, sr,
            )
        except Exception as exc:
            logger.exception("Smallest AI STT recognize failed: %s", exc)
            raise APIConnectionError() from exc
        duration_sec = sum(f.duration for f in frames)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language=lang, text=text.strip() or "")],
            recognition_usage=stt.RecognitionUsage(audio_duration=duration_sec),
        )

    def stream(
        self,
        *,
        language: Optional[str] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        sample_rate: Optional[int] = None,
    ) -> SmallestSTTSpeechStream:
        return SmallestSTTSpeechStream(
            stt_instance=self,
            conn_options=conn_options,
            sample_rate=sample_rate,
            language=language or self._language,
            api_key=self._api_key,
        )
