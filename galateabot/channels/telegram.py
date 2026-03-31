import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram import F

from galateabot.bus.events import InboundMessage, OutboundMessage
from galateabot.bus.queue import MessageBus

logger = logging.getLogger("telegram-channel")

class TelegramChannel:
    def __init__(self, token: str, bus: MessageBus):
        self._bot = Bot(token=token)
        self._dp = Dispatcher()
        self._bus = bus
        self._register_handlers()

    def _register_handlers(self):
        @self._dp.message(CommandStart())
        async def command_start_handler(message: types.Message):
            await message.answer(f"Hello, {message.from_user.full_name}! I am Galatea.")
            # Notify the agent loop about the new user/session
            await self._on_message(message)

        @self._dp.message(F.text)
        async def text_handler(message: types.Message):
            await self._on_message(message)

        @self._dp.message(F.photo)
        async def photo_handler(message: types.Message):
            # Handle photo (e.g., get the highest resolution file_id)
            photo = message.photo[-1]
            file_info = await self._bot.get_file(photo.file_id)
            file_url = f"https://api.telegram.org/file/bot{self._bot.token}/{file_info.file_path}"
            await self._on_message(message, media_url=file_url, media_type="photo")

        @self._dp.message(F.voice)
        async def voice_handler(message: types.Message):
             # Handle voice
            voice = message.voice
            file_info = await self._bot.get_file(voice.file_id)
            file_url = f"https://api.telegram.org/file/bot{self._bot.token}/{file_info.file_path}"
            await self._on_message(message, media_url=file_url, media_type="voice")

    async def _on_message(self, message: types.Message, media_url=None, media_type=None):
        inbound = InboundMessage(
            source="telegram",
            user_id=str(message.from_user.id),
            chat_id=str(message.chat.id),
            text=message.text or message.caption or "",
            media_url=media_url,
            media_type=media_type,
            raw_message=message
        )
        await self._bus.publish_inbound(inbound)

    async def start(self):
        logger.info("Starting Telegram Bot...")
        asyncio.create_task(self._listen_outbound())
        asyncio.create_task(self._listen_history_requests())
        await self._dp.start_polling(self._bot)

    async def _listen_outbound(self):
        async for msg in self._bus.subscribe_outbound("telegram"):
            try:
                await self._bot.send_message(
                    chat_id=msg.chat_id,
                    text=msg.text,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Failed to send message to Telegram: {e}")

    async def _listen_history_requests(self):
        async for req in self._bus.subscribe_history_requests("telegram"):
            logger.info(f"Processing history request for chat_id={req.chat_id}, limit={req.limit}")
            try:
                # Note: Telegram Bot API has limitations on fetching history. 
                # Usually bots only see messages sent to them while they are active.
                # However, for this implementation, we simulate fetching 'history' 
                # if we were logging it, or we could use user-bot APIs.
                # Here we just acknowledge the request for now.
                inbound = InboundMessage(
                    source="telegram",
                    user_id="system",
                    chat_id=req.chat_id,
                    text=f"[SYSTEM] History scan requested for {req.chat_id} (Limit: {req.limit}).",
                    payload={"type": "history_response", "request_id": req.request_id}
                )
                await self._bus.publish_inbound(inbound)
            except Exception as e:
                logger.error(f"Failed to process history request: {e}")
