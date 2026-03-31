"""
Voice agent with TTS decorator pattern: Rime or ElevenLabs selected per voice config.
Run: python voice_agent.py dev | console
Set VOICE=celeste (Rime) or VOICE=alex (ElevenLabs); default from VOICE_NAMES.
"""
import logging
import os
import random
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    metrics,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai, noise_cancellation, silero

from agent_config import VOICE_CONFIGS
from tts_providers import get_tts

load_dotenv()
logger = logging.getLogger("voice-agent")

VOICE_NAMES = list(VOICE_CONFIGS.keys())
VOICE = os.getenv("VOICE", random.choice(VOICE_NAMES))


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


class VoiceAssistant(Agent):
    def __init__(self, voice_key: str) -> None:
        config = VOICE_CONFIGS[voice_key]
        super().__init__(instructions=config["llm_prompt"])
        self.voice_key = voice_key


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    if VOICE not in VOICE_CONFIGS:
        logger.error("VOICE=%s not in VOICE_CONFIGS. Use one of: %s", VOICE, VOICE_NAMES)
        return

    config = VOICE_CONFIGS[VOICE]
    provider = config.get("provider", "rime").lower()
    logger.info(
        "Running voice agent: voice=%s provider=%s participant=%s",
        VOICE,
        provider,
        participant.identity,
    )

    voice_tts = get_tts(VOICE)

    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=voice_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=None,
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(VOICE),
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
        room_output_options=RoomOutputOptions(audio_enabled=True),
    )

    await session.say(config["intro_phrase"])


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
