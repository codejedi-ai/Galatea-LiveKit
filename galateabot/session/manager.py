from typing import Dict, List
from livekit.agents.llm import ChatMessage

class Session:
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.history: List[ChatMessage] = []

    def add_message(self, role: str, content: str):
        self.history.append(ChatMessage(role=role, content=content))
        # Keep history manageable (e.g., last 20 messages)
        if len(self.history) > 20:
            self.history = self.history[-20:]

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_session(self, chat_id: str) -> Session:
        if chat_id not in self._sessions:
            self._sessions[chat_id] = Session(chat_id)
        return self._sessions[chat_id]
