import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from galatea_livekit.utils.paths import PathManager

logger = logging.getLogger("galatea-history")

class HistoryManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or PathManager.get_db_path("history")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT,
                    chat_id TEXT,
                    user_id TEXT,
                    role TEXT,
                    text TEXT,
                    payload JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_entry(self, channel: str, chat_id: str, user_id: str, role: str, text: str, payload: Dict = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (channel, chat_id, user_id, role, text, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (channel, chat_id, user_id, role, text, json.dumps(payload or {}))
            )
            conn.commit()

    def get_history(self, channel: str, chat_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, text, payload, created_at FROM messages WHERE channel = ? AND chat_id = ? ORDER BY id DESC LIMIT ?",
                (channel, chat_id, limit)
            )
            rows = cursor.fetchall()
            # Reverse to get chronological order
            return [{"role": r[0], "text": r[1], "payload": json.loads(r[2]), "timestamp": r[3]} for r in reversed(rows)]
