from typing import List, Optional, Dict
from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .elevenlabs import ElevenLabsProvider
from .google import GoogleProvider

class ProviderManager:
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "elevenlabs": ElevenLabsProvider(),
            "google": GoogleProvider(),
        }

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        return self.providers.get(name.lower())

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())

    async def validate_all(self) -> Dict[str, bool]:
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.validate()
        return results

# Singleton instance
manager = ProviderManager()
