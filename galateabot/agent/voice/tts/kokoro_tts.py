"""
Kokoro TTS plugin for LiveKit agents.
Uses an OpenAI-compatible Kokoro server (e.g. Kokoro-FastAPI from Hugging Face).
See: https://github.com/remsky/Kokoro-FastAPI  (runs on port 8880 by default)
"""
import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

import httpx
import openai

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
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

TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1

TTSModels = Literal["tts-1", "kokoro"]
TTSVoices = Literal["af_heart", "af_bella", "af_sky"]


@dataclass
class KokoroTTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    speed: float


class KokoroTTS(tts.TTS):
    """TTS using Kokoro (OpenAI-compatible server, e.g. Kokoro-FastAPI)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8880/v1",
        api_key: str = "not-needed",
        model: TTSModels | str = "kokoro",
        voice: TTSVoices | str = "af_bella",
        speed: float = 1.0,
        client: Optional[openai.AsyncOpenAI] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )
        self._opts = KokoroTTSOptions(model=model, voice=voice, speed=speed)
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=30.0, write=5.0, pool=5.0),
                follow_redirects=True,
            ),
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KokoroTTSStream":
        return KokoroTTSStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._client,
        )


class KokoroTTSStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts_instance: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: KokoroTTSOptions,
        client: openai.AsyncOpenAI,
    ) -> None:
        super().__init__(tts=tts_instance, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._opts = opts

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )
        try:
            start_time = time.time()
            response = await self._client.audio.speech.create(
                model=self._opts.model,
                voice=self._opts.voice,
                input=self.input_text,
                response_format="pcm",
                speed=self._opts.speed,
            )
            if hasattr(response, "content"):
                data = response.content
            else:
                data = getattr(response, "body", b"") or b""
            if isinstance(data, bytes):
                for frame in audio_bstream.write(data):
                    self._event_ch.send_nowait(tts.SynthesizedAudio(frame=frame, request_id=request_id))
            for frame in audio_bstream.flush():
                self._event_ch.send_nowait(tts.SynthesizedAudio(frame=frame, request_id=request_id))
            logger.info("Kokoro TTS synthesis completed in %.1fms", (time.time() - start_time) * 1000)
        except openai.APITimeoutError as exc:
            raise APITimeoutError() from exc
        except openai.APIStatusError as exc:
            raise APIStatusError(
                getattr(exc, "message", str(exc)),
                status_code=getattr(exc, "status_code", 0),
                request_id=getattr(exc, "request_id", None),
                body=getattr(exc, "body", None),
            ) from exc
        except Exception as exc:
            raise APIConnectionError() from exc
