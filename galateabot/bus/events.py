from dataclasses import dataclass, field
from typing import Any, Optional, Dict

@dataclass
class InboundMessage:
    source: str
    user_id: str
    chat_id: str
    text: str
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    raw_message: Any = None
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutboundMessage:
    target: str
    user_id: str
    chat_id: str
    text: str
    media_url: Optional[str] = None
    media_type: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HistoryRequest:
    channel: str
    chat_id: str
    limit: int = 100
    request_id: str = ""
