from .base import BaseProvider, Model
import os
from typing import Optional

class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Anthropic", api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.models = [
            Model("claude-3-5-sonnet-latest", "Claude 3.5 Sonnet", ["text", "image"], ["text"], "anthropic", is_instruct=True, base="anthropic"),
            Model("claude-3-5-haiku-latest", "Claude 3.5 Haiku", ["text"], ["text"], "anthropic", is_instruct=True, base="anthropic"),
            Model("claude-3-opus-latest", "Claude 3 Opus", ["text", "image"], ["text"], "anthropic", is_instruct=True, base="anthropic"),
        ]

    async def validate(self) -> bool:
        return bool(self.api_key)
