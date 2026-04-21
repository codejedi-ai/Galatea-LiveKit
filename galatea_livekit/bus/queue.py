import asyncio
import sqlite3
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any
from .events import InboundMessage, OutboundMessage, HistoryRequest
from galatea_livekit.utils.paths import PathManager

logger = logging.getLogger("sqlite-bus")

class MessageBus:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or PathManager.get_db_path("galatea_bus")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Table for inbound messages
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inbound (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    user_id TEXT,
                    chat_id TEXT,
                    payload JSON,
                    processed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for outbound messages
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outbound (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target TEXT,
                    user_id TEXT,
                    chat_id TEXT,
                    payload JSON,
                    processed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for history requests (from AI to Channels)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel TEXT,
                    chat_id TEXT,
                    limit_count INTEGER,
                    request_id TEXT,
                    processed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    async def publish_inbound(self, msg: InboundMessage):
        def _insert():
            payload = {
                "text": msg.text,
                "media_url": msg.media_url,
                "media_type": msg.media_type,
                "payload": msg.payload
            }
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO inbound (source, user_id, chat_id, payload) VALUES (?, ?, ?, ?)",
                    (msg.source, msg.user_id, msg.chat_id, json.dumps(payload))
                )
                conn.commit()
        await asyncio.get_event_loop().run_in_executor(None, _insert)

    async def publish_outbound(self, msg: OutboundMessage):
        def _insert():
            payload = {
                "text": msg.text,
                "media_url": msg.media_url,
                "media_type": msg.media_type,
                "payload": msg.payload
            }
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO outbound (target, user_id, chat_id, payload) VALUES (?, ?, ?, ?)",
                    (msg.target, msg.user_id, msg.chat_id, json.dumps(payload))
                )
                conn.commit()
        await asyncio.get_event_loop().run_in_executor(None, _insert)

    async def publish_history_request(self, req: HistoryRequest):
        def _insert():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO history_requests (channel, chat_id, limit_count, request_id) VALUES (?, ?, ?, ?)",
                    (req.channel, req.chat_id, req.limit, req.request_id)
                )
                conn.commit()
        await asyncio.get_event_loop().run_in_executor(None, _insert)

    async def subscribe_inbound(self) -> AsyncGenerator[InboundMessage, None]:
        while True:
            msg = await self._get_next_inbound()
            if msg:
                yield msg
            else:
                await asyncio.sleep(0.1)

    async def _get_next_inbound(self) -> Optional[InboundMessage]:
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, source, user_id, chat_id, payload FROM inbound WHERE processed = 0 ORDER BY id LIMIT 1")
                row = cursor.fetchone()
                if row:
                    cursor.execute("UPDATE inbound SET processed = 1 WHERE id = ?", (row[0],))
                    conn.commit()
                    payload = json.loads(row[4])
                    return InboundMessage(
                        source=row[1],
                        user_id=row[2],
                        chat_id=row[3],
                        text=payload.get("text"),
                        media_url=payload.get("media_url"),
                        media_type=payload.get("media_type"),
                        payload=payload.get("payload", {})
                    )
                return None
        return await asyncio.get_event_loop().run_in_executor(None, _query)

    async def subscribe_outbound(self, channel: str) -> AsyncGenerator[OutboundMessage, None]:
        while True:
            msg = await self._get_next_outbound(channel)
            if msg:
                yield msg
            else:
                await asyncio.sleep(0.1)

    async def _get_next_outbound(self, channel: str) -> Optional[OutboundMessage]:
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, target, user_id, chat_id, payload FROM outbound WHERE processed = 0 AND target = ? ORDER BY id LIMIT 1", (channel,))
                row = cursor.fetchone()
                if row:
                    cursor.execute("UPDATE outbound SET processed = 1 WHERE id = ?", (row[0],))
                    conn.commit()
                    payload = json.loads(row[4])
                    return OutboundMessage(
                        target=row[1],
                        user_id=row[2],
                        chat_id=row[3],
                        text=payload.get("text"),
                        media_url=payload.get("media_url"),
                        media_type=payload.get("media_type"),
                        payload=payload.get("payload", {})
                    )
                return None
        return await asyncio.get_event_loop().run_in_executor(None, _query)

    async def subscribe_history_requests(self, channel: str) -> AsyncGenerator[HistoryRequest, None]:
        while True:
            req = await self._get_next_history_request(channel)
            if req:
                yield req
            else:
                await asyncio.sleep(0.5)

    async def _get_next_history_request(self, channel: str) -> Optional[HistoryRequest]:
        def _query():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, channel, chat_id, limit_count, request_id FROM history_requests WHERE processed = 0 AND channel = ? ORDER BY id LIMIT 1", (channel,))
                row = cursor.fetchone()
                if row:
                    cursor.execute("UPDATE history_requests SET processed = 1 WHERE id = ?", (row[0],))
                    conn.commit()
                    return HistoryRequest(
                        channel=row[1],
                        chat_id=row[2],
                        limit=row[3],
                        request_id=row[4]
                    )
                return None
        return await asyncio.get_event_loop().run_in_executor(None, _query)
