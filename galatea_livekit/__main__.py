# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "livekit-agents",
#     "livekit-plugins-elevenlabs",
#     "livekit-plugins-openai",
#     "livekit-plugins-anthropic",
#     "livekit-plugins-google",
#     "livekit-plugins-silero",
#     "livekit-plugins-noise-cancellation",
#     "python-dotenv",
#     "requests",
#     "aiohttp",
# ]
# ///

import asyncio
import logging
import os
import re
import sys
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
from livekit.plugins import silero, anthropic

# Ensure the project root (parent of 'galatea_livekit') is in sys.path so 'galatea_livekit' imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from galatea_livekit.providers import ElevenLabsTTS, ElevenLabsSTT
from galatea_livekit.bus.events import InboundMessage, OutboundMessage
from galatea_livekit.bus.queue import MessageBus
from galatea_livekit.utils.paths import PathManager

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
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

def load_workspace() -> dict:
    """Load SOUL, RULES, and PERSONALITY from the workspace."""
    user_root = PathManager.get_root()
    repo_workspace = Path(__file__).resolve().parent / "workspace"
    
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
    bus = MessageBus()
    chat_id = f"voice_{ctx.room.name}"

    # 3. Tools for Agent to interact with the Galatea Ecosystem (the galatea_livekit)
    @function_tool(description="Send a physical or system command to Natasha's galatea_livekit processing loop.")
    async def command_body(
        text: Annotated[str, "The command or intent to send to the galatea_livekit"]
    ) -> str:
        logger.info(f"Tool Call: command_body text='{text}'")
        msg = InboundMessage(
            source="voice",
            user_id="user",
            chat_id=chat_id,
            text=text
        )
        await bus.publish_inbound(msg)
        return "Command sent to galatea_livekit bus."

    # 4. Initialize Agent & Session
    agent = GalateaVoiceAgent(
        instructions=system_prompt,
        tools=[command_body],
    )

    # Use ElevenLabs for BOTH TTS and STT as requested
    voice_id = os.getenv("ELEVEN_VOICE_ID", "95XPUDALaQL1LY3I023E")

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=ElevenLabsSTT(), # Native ElevenLabs Scribe STT
        llm=anthropic.LLM(model="claude-haiku-4-5"),
        tts=ElevenLabsTTS(voice_id=voice_id),
        tools=[command_body]
    )

    # 5. Listen for Outbound Queue (Responses from Galatea Bot)
    async def _listen_outbound():
        async for msg in bus.subscribe_outbound("voice"):
            if msg.chat_id == chat_id:
                logger.info(f"Speaking outbound message from bus: {msg.text}")
                await session.say(msg.text)

    asyncio.create_task(_listen_outbound())

    # 6. Bridge Speech Commitment directly to Inbound Queue
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
    # Default to console if no args
    if len(sys.argv) == 1:
        sys.argv.append("console")
    cli.run_app(server)
