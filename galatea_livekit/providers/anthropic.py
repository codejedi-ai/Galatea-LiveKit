from .base import BaseProvider, Model
import os
from typing import Optional

class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Anthropic", api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.models = [
            Model("claude-sonnet-4-6", "Claude 4.6 Sonnet", ["text", "image"], ["text"], "anthropic", is_instruct=True, base="anthropic"),
            Model("claude-haiku-4-5", "Claude 4.5 Haiku", ["text"], ["text"], "anthropic", is_instruct=True, base="anthropic"),
            Model("claude-opus-4-7", "Claude 4.7 Opus", ["text", "image"], ["text"], "anthropic", is_instruct=True, base="anthropic"),
        ]

    async def validate(self) -> bool:
        return bool(self.api_key)
