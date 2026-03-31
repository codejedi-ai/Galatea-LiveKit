"""
Silero STT plugin for LiveKit agents.
Runs locally in-process (no API): uses snakers4/silero-models (silero_stt) via torch.hub.
Requires: torch, torchaudio, omegaconf.
"""
import asyncio
import logging
import tempfile
from pathlib import Path
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

SILERO_STT_SAMPLE_RATE = 16000


def _transcribe_sync(language: str, pcm_bytes: bytes, sample_rate: int) -> str:
    """Run Silero STT in a thread; returns transcribed text. Audio must be 16 kHz mono."""
    import torch
    device = torch.device("cpu")
    model, decoder, silero_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        language=language,
        device=device,
    )
    (read_batch, _split_into_batches, read_audio, prepare_model_input) = silero_utils
    # Silero expects 16 kHz normalized (-1..1). Write to temp wav for read_audio.
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    tensor = torch.from_numpy(audio_float).unsqueeze(0)
    if sample_rate != SILERO_STT_SAMPLE_RATE:
        import torchaudio
        tensor = torchaudio.functional.resample(
            tensor, sample_rate, SILERO_STT_SAMPLE_RATE
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        import torchaudio
        torchaudio.save(wav_path, tensor, SILERO_STT_SAMPLE_RATE)
        batches = [[wav_path]]
        input_tensor = prepare_model_input(read_batch(batches[0]), device=device)
        output = model(input_tensor)
        text_parts = []
        for example in output:
            text_parts.append(decoder(example.cpu()))
        return " ".join(str(t) for t in text_parts).strip()
    finally:
        Path(wav_path).unlink(missing_ok=True)


class SileroSTTSpeechStream(stt.SpeechStream):
    """RecognizeStream that buffers audio, runs Silero STT on flush."""

    def __init__(
        self,
        *,
        stt_instance: "SileroSTT",
        conn_options: APIConnectOptions,
        sample_rate: Optional[int] = None,
        language: str = "en",
    ) -> None:
        super().__init__(
            stt=stt_instance,
            conn_options=conn_options,
            sample_rate=sample_rate if sample_rate is not None else NOT_GIVEN,
        )
        self._language = language

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
                            self._language,
                            pcm,
                            sr,
                        )
                    except Exception as exc:
                        logger.exception("Silero STT failed: %s", exc)
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
            logger.exception("Silero STT stream error: %s", exc)
            raise


class SileroSTT(stt.STT):
    """STT using Silero models (snakers4/silero-models silero_stt)."""

    def __init__(self, language: str = "en") -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self._language = language

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
            text = await loop.run_in_executor(None, _transcribe_sync, lang, pcm, sr)
        except Exception as exc:
            logger.exception("Silero STT recognize failed: %s", exc)
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
    ) -> SileroSTTSpeechStream:
        return SileroSTTSpeechStream(
            stt_instance=self,
            conn_options=conn_options,
            sample_rate=sample_rate,
            language=language or self._language,
        )
