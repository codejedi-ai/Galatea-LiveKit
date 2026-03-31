"""
Silero TTS plugin for LiveKit agents.
Runs locally in-process (no API): uses snakers4/silero-models (silero_tts) via torch.hub.
Requires: torch, torchaudio, omegaconf
"""
import asyncio
import logging
import time
from typing import Optional

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)

SILERO_TTS_SAMPLE_RATE = 16000
TTS_CHANNELS = 1


def _synthesize_sync(
    language: str,
    speaker: str,
    text: str | list,
) -> tuple[bytes, int]:
    """Run Silero TTS in a thread; returns (pcm_bytes, sample_rate)."""
    import numpy as np
    import torch
    # Normalize to a single string (framework may pass list of segments)
    if isinstance(text, list):
        text = " ".join(str(t) for t in text) if text else ""
    text = str(text).strip() or " "
    device = torch.device("cpu")
    model, symbols, sample_rate, _example_text, apply_tts = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language=language,
        speaker=speaker,
    )
    model = model.to(device)
    # apply_tts expects texts = list of strings (one string per utterance)
    audio = apply_tts(
        texts=[text],
        model=model,
        sample_rate=sample_rate,
        symbols=symbols,
        device=device,
    )
    # audio: list of tensors or tensor; normalize to (samples,) float32
    if isinstance(audio, (list, tuple)):
        audio = audio[0] if audio else torch.zeros(0)
    if audio.dim() > 1:
        audio = audio.squeeze()
    arr = audio.cpu().float().numpy()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)
    return pcm.tobytes(), int(sample_rate)


class SileroTTSStream(tts.ChunkedStream):
    """ChunkedStream that runs Silero TTS in a thread and emits via output_emitter."""

    def __init__(
        self,
        *,
        tts_instance: "SileroTTS",
        input_text: str,
        conn_options: APIConnectOptions,
        language: str,
        speaker: str,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._language = language
        self._speaker = speaker

    async def _run(self, output_emitter) -> None:
        request_id = utils.shortuuid()
        loop = asyncio.get_event_loop()
        try:
            start_time = time.time()
            pcm_bytes, sample_rate = await loop.run_in_executor(
                None,
                _synthesize_sync,
                self._language,
                self._speaker,
                self.input_text,
            )
        except Exception as exc:
            logger.exception("Silero TTS failed: %s", exc)
            raise APIConnectionError() from exc
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=sample_rate,
            num_channels=TTS_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm_bytes)
        logger.info(
            "Silero TTS completed in %.1fms (lang=%s speaker=%s)",
            (time.time() - start_time) * 1000,
            self._language,
            self._speaker,
        )


class SileroTTS(tts.TTS):
    """TTS using Silero models (snakers4/silero-models silero_tts)."""

    def __init__(
        self,
        language: str = "en",
        speaker: str = "lj_16khz",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SILERO_TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )
        self._language = language
        self._speaker = speaker

    @property
    def model(self) -> str:
        """Model identifier for metrics (e.g. silero_tts-en-lj_16khz)."""
        return f"silero_tts-{self._language}-{self._speaker}"

    @property
    def provider(self) -> str:
        """Provider name for metrics."""
        return "silero"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SileroTTSStream:
        return SileroTTSStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
            language=self._language,
            speaker=self._speaker,
        )
