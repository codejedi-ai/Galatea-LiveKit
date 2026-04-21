import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    FunctionTool,
    JobContext,
    ModelSettings,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import silero
from galateabot.providers import ElevenLabsTTS, ElevenLabsSTT
from galateabot.bus.events import InboundMessage, OutboundMessage
from galateabot.bus.queue import MessageBus
from galateabot.utils.paths import PathManager

# Configure logging
logger = logging.getLogger("galatea-voice-agent")
logger.setLevel(logging.INFO)

load_dotenv()

class GalateaVoiceAgent(Agent):
    def __init__(self, instructions: str, tools: list[FunctionTool]) -> None:
        super().__init__(instructions=instructions, tools=tools)

    async def llm_node(
        self, chat_ctx: ChatContext, tools: list[FunctionTool], model_settings: ModelSettings
    ):
        # Voice agent reasoning node
        return await Agent.default.llm_node(self, chat_ctx, tools, model_settings)

def load_workspace() -> dict:
    """Load SOUL, RULES, and PERSONALITY from the workspace."""
    user_root = PathManager.get_root()
    repo_workspace = Path("galateabot/workspace")
    
    def _load_file(name: str) -> str:
        user_file = user_root / name
        repo_file = repo_workspace / name
        if user_file.exists():
            return user_file.read_text(encoding="utf-8")
        if repo_file.exists():
            return repo_file.read_text(encoding="utf-8")
        return ""

    return {
        "soul": _load_file("SOUL.md"),
        "rules": _load_file("RULES.md"),
        "personality": _load_file("PERSONALITY.md"),
    }

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"Starting Galatea Voice Session: {ctx.room.name}")
    
    # 1. Load Workspace
    workspace = load_workspace()
    system_prompt = f"{workspace['soul']}\n\n{workspace['rules']}\n\n{workspace['personality']}"

    # 2. Data Structures: Queues for Bus Integration
    # 'in_bus' (InboundMessage) triggered via tool calls or speech commitment
    # 'out_bus' (OutboundMessage) subscribed to for spoken output
    bus = MessageBus()
    chat_id = f"voice_{ctx.room.name}"

    # 3. Tools for Agent to interact with the Galatea Ecosystem
    @function_tool(description="Send a command or message to the main Galatea Bot processing loop.")
    async def call_galatea_bot(
        text: Annotated[str, "The command or message text to send to the bot"]
    ) -> str:
        logger.info(f"Tool Call: call_galatea_bot text='{text}'")
        msg = InboundMessage(
            source="voice",
            user_id="user",
            chat_id=chat_id,
            text=text
        )
        await bus.publish_inbound(msg)
        return "Message sent to Galatea Bot bus."

    # 4. Initialize Agent & Session
    agent = GalateaVoiceAgent(
        instructions=system_prompt,
        tools=[call_galatea_bot],
    )

    # Use ElevenLabs for BOTH TTS and STT as requested
    voice_id = os.getenv("ELEVEN_VOICE_ID", "95XPUDALaQL1LY3I023E")
    
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=ElevenLabsSTT(), # Custom Consolidated ElevenLabs STT
        llm=inference.LLM("openai/gpt-4o-mini"),
        tts=ElevenLabsTTS(voice_id=voice_id),
        tools=[call_galatea_bot]
    )

    # 5. Listen for Outbound Queue (Responses from Galatea Bot)
    async def _listen_outbound():
        async for msg in bus.subscribe_outbound("voice"):
            if msg.chat_id == chat_id:
                logger.info(f"Speaking outbound message from bus: {msg.text}")
                await session.say(msg.text)

    asyncio.create_task(_listen_outbound())

    # 6. Bridge Speech Commitment directly to Inbound Queue (Optional, if tool is preferred)
    @session.on("user_speech_committed")
    def _on_speech(msg):
        text = msg.text.strip()
        if text:
            asyncio.create_task(bus.publish_inbound(InboundMessage(
                source="voice",
                user_id="user",
                chat_id=chat_id,
                text=text
            )))

    await session.start(agent, room=ctx.room)
    
    # Natasha Greeting
    await session.say("Hey there. It's Natasha. I'm ready to help you manage everything. What should we do first?")

if __name__ == "__main__":
    cli.run_app(server)
