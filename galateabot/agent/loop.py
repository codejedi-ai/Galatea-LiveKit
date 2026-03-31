import asyncio
import logging
import os
import re
from pathlib import Path
from livekit.plugins import google
from livekit.agents.llm import ChatMessage, ChatContext, FunctionContext
from inflection_llm import InflectionLLM
from galateabot.bus.events import InboundMessage, OutboundMessage
from galateabot.bus.queue import MessageBus
from galateabot.session.manager import SessionManager
from galateabot.config.manager import ConfigManager
from galateabot.utils.paths import PathManager

from galateabot.agent.tools.fs_tools import read_file, write_file, list_directory, create_directory
from galateabot.agent.tools.bucket_tools import bucket_put, bucket_get, bucket_list
from galateabot.agent.tools.message_tools import send_message
from galateabot.agent.tools.history_tools import query_local_history, request_channel_history
from galateabot.session.history import HistoryManager

logger = logging.getLogger("agent-loop")

class AgentLoop:
    def __init__(self, bus: MessageBus):
        self._bus = bus
        self._sessions = SessionManager()
        self._history = HistoryManager()
        
        # Load from new config structure
        llm_model = ConfigManager.get("llm", "text.llm_model", "gemini-3.1-pro-preview")
        emotional_model = ConfigManager.get("llm", "text.emotional_layer_model", "Pi-3.1")
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            self._llm = google.LLM(model=llm_model)
        else:
            self._llm = None
            logger.warning("GOOGLE_API_KEY not found. MOCK MODE.")

        self._emotional_layer = InflectionLLM(model=emotional_model)
        
        # Unified Workspace Loading
        # Priority: .galatea user root > repo templates
        user_root = PathManager.get_root()
        repo_workspace = Path("galateabot/workspace")
        
        def _load_instruction_file(name: str) -> str:
            user_file = user_root / name
            repo_file = repo_workspace / name
            if user_file.exists():
                return user_file.read_text(encoding="utf-8")
            if repo_file.exists():
                # Copy to user root if it doesn't exist there yet for easier user customization
                import shutil
                shutil.copy(repo_file, user_file)
                return repo_file.read_text(encoding="utf-8")
            return ""

        self._soul = _load_instruction_file("SOUL.md")
        self._rules = _load_instruction_file("RULES.md")
        self._personality = _load_instruction_file("PERSONALITY.md")
        
        # Extract baseline vibe from PERSONALITY.md
        vibe_match = re.search(r"## BASELINE EMOTIONAL VIBE\n(.*?)\n", self._personality, re.DOTALL)
        self._baseline_vibe = vibe_match.group(1).strip() if vibe_match else "empathetic and calm"
        
        # Setup Function Context
        self._fnc_ctx = FunctionContext()
        self._fnc_ctx.add_callable(read_file)
        self._fnc_ctx.add_callable(write_file)
        self._fnc_ctx.add_callable(list_directory)
        self._fnc_ctx.add_callable(create_directory)
        self._fnc_ctx.add_callable(bucket_put)
        self._fnc_ctx.add_callable(bucket_get)
        self._fnc_ctx.add_callable(bucket_list)
        self._fnc_ctx.add_callable(send_message)
        self._fnc_ctx.add_callable(query_local_history)
        self._fnc_ctx.add_callable(request_channel_history)

    async def start(self):
        logger.info("Starting Galatea Agent Loop...")
        async for msg in self._bus.subscribe_inbound():
            asyncio.create_task(self._process_message(msg))

    async def _process_message(self, msg: InboundMessage):
        # Log inbound message
        self._history.add_entry(
            channel=msg.source,
            chat_id=msg.chat_id,
            user_id=msg.user_id,
            role="user",
            text=msg.text,
            payload=msg.payload
        )

        session = self._sessions.get_session(msg.chat_id)
        
        if not self._llm:
            response_text = f"[MOCK] I received: '{msg.text}'"
        else:
            # 1. Emotional Layer (Inflection)
            current_vibe = self._baseline_vibe
            try:
                inflection_ctx = ChatContext(messages=[
                    ChatMessage(role="system", content=f"Analyze the emotional vibe of the user's message. Baseline personality: {self._personality}"),
                    ChatMessage(role="user", content=msg.text)
                ])
                inflection_stream = self._emotional_layer.chat(chat_ctx=inflection_ctx)
                vibe_text = ""
                async for chunk in inflection_stream:
                     if chunk.choices[0].delta.content:
                         vibe_text += chunk.choices[0].delta.content
                if vibe_text:
                    current_vibe = vibe_text.strip()
            except Exception as e:
                logger.error(f"Inflection failed: {e}")

            # 2. Build Unified Context
            system_prompt = f"{self._soul}\n\n{self._rules}\n\n{self._personality}\n\nCURRENT RESONANCE: {current_vibe}"
            session.add_message("user", msg.text)
            
            thinking_level = ConfigManager.get("llm", "text.thinking_level", "medium")
            
            try:
                chat_ctx = ChatContext(messages=[ChatMessage(role="system", content=system_prompt)] + session.history)
                stream = self._llm.chat(
                    chat_ctx=chat_ctx,
                    fnc_ctx=self._fnc_ctx,
                    extra_kwargs={"thinking_config": {"thinking_level": thinking_level}}
                )
                
                response_text = ""
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        response_text += content
            except Exception as e:
                logger.error(f"Error: {e}")
                response_text = f"Dissonance detected: {e}"

        # 5. Send Response
        outbound = OutboundMessage(
            target=msg.source,
            user_id=msg.user_id,
            chat_id=msg.chat_id,
            text=response_text
        )
        
        # Log outbound message
        self._history.add_entry(
            channel=msg.source,
            chat_id=msg.chat_id,
            user_id="galatea",
            role="assistant",
            text=response_text
        )

        session.add_message("assistant", response_text)
        await self._bus.publish_outbound(outbound)
