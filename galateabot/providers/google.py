from .base import BaseProvider, Model
import os
from typing import Optional

class GoogleProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Google", api_key or os.getenv("GOOGLE_API_KEY"))
        self.models = [
            Model("gemini-1.5-pro", "Gemini 1.5 Pro", ["text", "image", "video", "voice"], ["text"], "google", is_instruct=True, base="gemini"),
            Model("gemini-1.5-flash", "Gemini 1.5 Flash", ["text", "image", "video", "voice"], ["text"], "google", is_instruct=True, base="gemini"),
        ]

    async def validate(self) -> bool:
        return bool(self.api_key)
