from .base import BaseProvider, Model
import os
from typing import Optional

class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("OpenAI", api_key or os.getenv("OPENAI_API_KEY"))
        self.models = [
            Model("gpt-4o", "GPT-4o", ["text", "image"], ["text"], "openai", is_instruct=True, base="openai"),
            Model("gpt-4o-mini", "GPT-4o Mini", ["text", "image"], ["text"], "openai", is_instruct=True, base="openai"),
            Model("dall-e-3", "DALL-E 3", ["text"], ["image"], "openai", is_instruct=False, base="openai"),
            Model("whisper-1", "Whisper-1", ["voice"], ["text"], "openai", is_instruct=False, base="openai"),
            Model("tts-1", "TTS-1", ["text"], ["voice"], "openai", is_instruct=False, base="openai"),
        ]

    async def validate(self) -> bool:
        return bool(self.api_key)
