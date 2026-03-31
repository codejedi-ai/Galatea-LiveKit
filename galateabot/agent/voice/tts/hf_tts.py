"""
Hugging Face TTS plugin for LiveKit agents.
Runs locally in-process (no API): uses transformers with models from the Hugging Face Hub.
Requires: pip install transformers torch
"""
import asyncio
import logging
import time
from typing import Optional

import numpy as np
import transformers  # Hugging Face model library; pipeline() used in _synthesize_sync

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

# SpeechT5 / common TTS output rate; we resample if needed
DEFAULT_SAMPLE_RATE = 16000
TTS_CHANNELS = 1


def _synthesize_sync(model_id: str, text: str, speaker_id: Optional[int] = None) -> tuple[bytes, int]:
    """Run HF TTS in a thread; returns (pcm_bytes, sample_rate). Uses Hugging Face model library."""
    # Prefer text-to-speech; some models use text-to-audio
    try:
        pipe = transformers.pipeline("text-to-speech", model=model_id, device=-1)
    except Exception:
        pipe = transformers.pipeline("text-to-audio", model=model_id, device=-1)
    kwargs = {} if speaker_id is None else {"speaker_id": speaker_id}
    out = pipe(text, **kwargs)
    # pipeline returns {"audio": ndarray, "sampling_rate": int} or similar
    if isinstance(out, dict):
        audio = out.get("audio", out.get("output", None))
        sr = out.get("sampling_rate", out.get("sample_rate", DEFAULT_SAMPLE_RATE))
    else:
        audio = out
        sr = DEFAULT_SAMPLE_RATE
    if audio is None:
        raise ValueError("TTS pipeline returned no audio")
    audio = np.asarray(audio)
    if audio.dtype != np.float32:
        if audio.dtype in (np.float64,):
            audio = audio.astype(np.float32)
        else:
            audio = audio.astype(np.float32) / (np.iinfo(audio.dtype).max or 1)
    # Mono: flatten if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Float [-1,1] -> 16-bit PCM
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm.tobytes(), int(sr)


class HFTTSStream(tts.ChunkedStream):
    """ChunkedStream that runs Hugging Face TTS in a thread and emits frames."""

    def __init__(
        self,
        *,
        tts_instance: "HFTTS",
        input_text: str,
        conn_options: APIConnectOptions,
        model_id: str,
        speaker_id: Optional[int],
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._model_id = model_id
        self._speaker_id = speaker_id

    async def _run(self, output_emitter) -> None:
        request_id = utils.shortuuid()
        loop = asyncio.get_event_loop()
        try:
            start_time = time.time()
            pcm_bytes, sample_rate = await loop.run_in_executor(
                None,
                _synthesize_sync,
                self._model_id,
                self.input_text,
                self._speaker_id if self._speaker_id is not None else None,
            )
        except Exception as exc:
            logger.exception("HF TTS failed: %s", exc)
            raise APIConnectionError() from exc
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=sample_rate,
            num_channels=TTS_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm_bytes)
        logger.info(
            "HF TTS synthesis completed in %.1fms (model=%s)",
            (time.time() - start_time) * 1000,
            self._model_id,
        )


class HFTTS(tts.TTS):
    """TTS using Hugging Face transformers (text-to-speech pipeline)."""

    def __init__(
        self,
        model: str = "microsoft/speecht5_tts",
        speaker_id: Optional[int] = 0,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )
        self._model_id = model
        self._speaker_id = speaker_id

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> HFTTSStream:
        return HFTTSStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
            model_id=self._model_id,
            speaker_id=self._speaker_id,
        )
