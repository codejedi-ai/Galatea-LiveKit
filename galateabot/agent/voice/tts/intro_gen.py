"""
Generate a short, casual first-line greeting using a small/fast model:
- Inflection Pi (e.g. Pi-3.1) via Inflection API — default, fast and small
- Or local LLM (e.g. Phi via Ollama) when model is not a Pi-* model
"""
import logging
import os

import aiohttp
from openai import AsyncOpenAI

logger = logging.getLogger("voice-agent")

INFLECTION_API_URL = "https://api.inflection.ai/external/api/inference"
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
# Default to Inflection Pi-3.1 (small, fast) for greeting
DEFAULT_MODEL = os.getenv("LLM_MODEL", "Pi-3.1")


def _is_inflection_model(model: str) -> bool:
    return model.startswith("Pi-") or model.lower() in ("inflection", "pi")


async def _generate_intro_inflection(
    prompt: str,
    *,
    model: str = "Pi-3.1",
) -> str | None:
    key = os.getenv("INFLECTION_AI_KEY")
    if not key:
        logger.warning("Intro generation: INFLECTION_AI_KEY not set, cannot use Inflection")
        return None
    context_data = [{"text": prompt.strip(), "type": "Human"}]
    payload = {"context": context_data, "config": model}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(INFLECTION_API_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.warning("Intro generation (Inflection) failed: %s - %s", resp.status, err_text)
                    return None
                data = await resp.json()
    except Exception as e:
        logger.warning("Intro generation (Inflection) failed: %s", e)
        return None
    text = data.get("text") or data.get("output") or data.get("response") or ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    text = (text or "").strip()
    return " ".join(text.split()) if text else None


async def _generate_intro_ollama(
    prompt: str,
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY") or "ollama"
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return " ".join(text.split()) if text else None
    except Exception as e:
        logger.warning("Intro generation (Ollama) failed %s: %s", base_url, e)
        return None


async def generate_intro(
    prompt: str,
    *,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.9,
    max_tokens: int = 80,
) -> str | None:
    """
    Generate one short greeting. Uses Inflection Pi (e.g. Pi-3.1) when model
    is Pi-* or 'inflection'; otherwise uses local LLM (Ollama).
    Returns the generated text or None on failure.
    """
    model = model or DEFAULT_MODEL
    if _is_inflection_model(model):
        return await _generate_intro_inflection(prompt, model=model)
    base_url = base_url or DEFAULT_BASE_URL
    return await _generate_intro_ollama(
        prompt,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
