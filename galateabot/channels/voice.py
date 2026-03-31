import asyncio
import logging
from pathlib import Path
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    VoiceAssistant,
)
from livekit.plugins import openai, silero, rime
from galateabot.bus.events import InboundMessage, OutboundMessage
from galateabot.bus.queue import MessageBus
from galateabot.config.manager import ConfigManager
from galateabot.agent.voice.vad.vad import load_vad

logger = logging.getLogger("voice-channel")

def prewarm_voice(proc: JobProcess):
    """Prewarm the VAD model for the voice channel."""
    proc.userdata["vad"] = load_vad()

class VoiceChannel:
    def __init__(self, bus: MessageBus):
        self._bus = bus
        self._assistant = None
        self._chat_id = None

    async def run_worker(self, ctx: JobContext):
        """Unified Voice Channel entrypoint."""
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        participant = await ctx.wait_for_participant()
        self._chat_id = f"voice_{participant.identity}"

        logger.info(f"Voice Channel linked for chat_id={self._chat_id}")

        # 1. Setup Modular STT/TTS
        stt_model = ConfigManager.get("stt", "model", "whisper-1")
        tts_model = ConfigManager.get("tts", "model", "arcana")
        voice_name = ConfigManager.get("tts", "voice", "celeste")

        # Start Listening for outbound bus messages to speak back
        asyncio.create_task(self._listen_outbound_bus())

        # 2. Bridge Voice -> STT -> Bus
        class VoiceToBusLLM:
            """A dummy LLM that does nothing. The real reasoning happens in AgentLoop via the Bus."""
            def chat(self, *args, **kwargs):
                return None

        # Use the prewarmed VAD if available
        vad = ctx.proc.userdata.get("vad") or load_vad()

        self._assistant = VoiceAssistant(
            vad=vad,
            stt=openai.STT(model=stt_model),
            tts=rime.TTS(model=tts_model, speaker=voice_name),
            llm=VoiceToBusLLM()
        )

        @self._assistant.on("user_speech_committed")
        def _on_user_speech(msg):
            # Capture the transcribed text and send it to the Inbound Bus
            text = msg.text.strip()
            if text:
                logger.info(f"Voice Channel Inbound: {text}")
                asyncio.create_task(self._bus.publish_inbound(InboundMessage(
                    source="voice",
                    user_id=participant.identity,
                    chat_id=self._chat_id,
                    text=text
                )))

        self._assistant.start(ctx.room, participant)
        
        while ctx.room.is_connected:
            await asyncio.sleep(1)

    async def _listen_outbound_bus(self):
        """Listen for Outbound messages from the Agent Loop and speak them."""
        async for msg in self._bus.subscribe_outbound("voice"):
            if msg.chat_id == self._chat_id and self._assistant:
                logger.info(f"Voice Channel Outbound: {msg.text}")
                # Use the assistant's 'say' method to speak back to the room
                await self._assistant.say(msg.text)
