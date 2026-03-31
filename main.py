import asyncio
import logging
import os
import signal
from dotenv import load_dotenv

from galateabot.channels.telegram import TelegramChannel
from galateabot.channels.cli import CLIChannel
from galateabot.channels.voice import VoiceChannel, prewarm_voice
from galateabot.bus.queue import MessageBus
from galateabot.agent.loop import AgentLoop

# LiveKit Agent Worker imports
from livekit.agents import WorkerOptions, Worker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("galateabot.log"),
        logging.StreamHandler()
    ]
)

# Global bus instance for tools/channels to access
GLOBAL_BUS = None

async def main():
    global GLOBAL_BUS
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    bus = MessageBus()
    GLOBAL_BUS = bus
    
    # 1. Initialize Channels
    channels = []
    
    # CLI Channel
    cli = CLIChannel(bus)
    channels.append(cli.start())
    
    # Telegram Channel
    if token:
        telegram = TelegramChannel(token, bus)
        channels.append(telegram.start())
    else:
        logging.warning("TELEGRAM_BOT_TOKEN not found. Telegram channel disabled.")

    # Voice Channel (LiveKit Bridge)
    voice_channel = VoiceChannel(bus)
    
    # 2. Initialize Agent Loop (Core Reasoning)
    agent_loop = AgentLoop(bus)
    
    # 3. Initialize LiveKit Worker
    voice_worker = Worker(
        WorkerOptions(
            entrypoint_fnc=voice_channel.run_worker,
            prewarm_fnc=prewarm_voice,
        )
    )

    logging.info("GalateaBot is awakening (CLI + Telegram + Voice Channel)...")

    try:
        await asyncio.gather(
            agent_loop.start(),
            voice_worker.run(),
            *channels
        )
    except Exception as e:
        logging.error(f"Critical error: {e}")
    finally:
        await shutdown(voice_worker)

async def shutdown(worker: Worker):
    logging.info("Galatea is going to sleep...")
    await worker.drain()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
