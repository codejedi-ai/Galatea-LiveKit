"""
Inflection AI tool: call Inflection (e.g. Pi) for a reply. Available to all agents.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import aiohttp

logger = logging.getLogger("inflection-tool")

INFLECTION_API_URL = "https://api.inflection.ai/external/api/inference"


async def get_inflection_response(
    user_message: str,
    *,
    system_instruction: Optional[str] = None,
    model: str = "Pi-3.1",
    api_key: Optional[str] = None,
) -> str:
    """
    Get a reply from Inflection AI (e.g. Pi) for the given user message.
    Optional system_instruction is prepended as context.
    """
    key = api_key or os.getenv("INFLECTION_AI_KEY")
    if not key:
        return "Inflection AI is not configured (set INFLECTION_AI_KEY)."

    context_data: List[dict[str, str]] = []
    if system_instruction:
        context_data.append({"text": f"System Instructions: {system_instruction}", "type": "Human"})
    context_data.append({"text": str(user_message).strip(), "type": "Human"})

    payload: dict[str, Any] = {"context": context_data, "config": model}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(INFLECTION_API_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                logger.error("Inflection API error: %s - %s", resp.status, err_text)
                return "I could not get a response from Inflection right now."
            try:
                data = await resp.json()
            except Exception as e:
                logger.error("Inflection response parse error: %s", e)
                return "I could not read the Inflection response."
    text = data.get("text") or data.get("output") or data.get("response") or ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    return (text or "I have nothing to add.").strip()
