"""
Hugging Face LLM plugin for LiveKit agents.
Runs locally in-process (no API): uses transformers (text-generation pipeline) with models from the Hugging Face Hub.
Requires: pip install transformers torch
"""
import asyncio
import logging
from typing import Optional

import transformers  # Hugging Face model library; pipeline() used in _generate_sync

from livekit.agents import utils
from livekit.agents.llm import (
    LLM,
    LLMStream,
    ChatChunk,
    ChoiceDelta,
    CompletionUsage,
)
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)


def _generate_sync(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Run HF text generation in a thread; returns generated text."""
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        device=-1,
    )
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    if not out or not isinstance(out, list):
        return ""
    gen = out[0]
    if isinstance(gen, dict) and "generated_text" in gen:
        full = gen["generated_text"]
        # Remove the prompt from the start if the model echoes it
        if full.startswith(prompt):
            return full[len(prompt) :].strip()
        return full.strip()
    return str(gen).strip()


class HFLLMStream(LLMStream):
    """LLMStream that runs Hugging Face generate in a thread and pushes ChatChunks."""

    def __init__(
        self,
        llm: "HFLLM",
        *,
        chat_ctx,
        tools: list,
        conn_options: APIConnectOptions,
        model_id: str,
        max_new_tokens: int,
        temperature: Optional[float],
    ) -> None:
        super().__init__(llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature or 0.7

    async def _run(self) -> None:
        from livekit.agents._exceptions import APIConnectionError
        # Build prompt from chat context (simple concatenation for compatibility)
        parts = []
        for msg in self._chat_ctx.messages():
            role = getattr(msg, "role", None) or "user"
            content = getattr(msg, "content", None) or ""
            if isinstance(content, list):
                content = " ".join(str(c) for c in content if isinstance(c, str))
            parts.append(f"{role.capitalize()}: {content}")
        prompt = "\n".join(parts) + "\nAssistant:"
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                _generate_sync,
                self._model_id,
                prompt,
                self._max_new_tokens,
                self._temperature,
            )
        except Exception as exc:
            logger.exception("HF LLM failed: %s", exc)
            raise APIConnectionError() from exc
        request_id = utils.shortuuid()
        # Approximate token counts for usage
        num_tokens = max(1, len(text.split()) * 2)  # rough
        self._event_ch.send_nowait(
            ChatChunk(
                id=request_id,
                delta=ChoiceDelta(role="assistant", content=text),
                usage=CompletionUsage(
                    completion_tokens=num_tokens,
                    prompt_tokens=max(1, len(prompt.split()) * 2),
                    total_tokens=num_tokens + max(1, len(prompt.split()) * 2),
                ),
            )
        )


class HFLLM(LLM):
    """LLM using Hugging Face transformers (text-generation pipeline)."""

    def __init__(
        self,
        model: str = "distilgpt2",
        max_new_tokens: int = 256,
        temperature: Optional[float] = 0.7,
    ) -> None:
        super().__init__()
        self._model_id = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "huggingface"

    def chat(
        self,
        *,
        chat_ctx,
        tools: Optional[list] = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> HFLLMStream:
        return HFLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            model_id=self._model_id,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
