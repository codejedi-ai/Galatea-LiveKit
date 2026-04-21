from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

@dataclass
class Model:
    id: str
    name: str
    input_media: List[str]  # ["text", "image", "voice", "video"]
    output_media: List[str] # ["text", "image", "voice", "video"]
    provider: str
    is_instruct: bool = False # Whether it supports tool calling / functions
    base: str = "custom"      # "openai", "anthropic", "gemini", "custom", etc.
    is_deprecated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseProvider(ABC):
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.models: List[Model] = []

    @abstractmethod
    async def validate(self) -> bool:
        pass

    def get_model(self, model_id: str) -> Optional[Model]:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def list_models(self) -> List[Model]:
        return self.models
