import asyncio
import logging
import sys
from galatea_livekit.bus.events import InboundMessage, OutboundMessage
from galatea_livekit.bus.queue import MessageBus

logger = logging.getLogger("cli-channel")

class CLIChannel:
    def __init__(self, bus: MessageBus):
        self._bus = bus
        self._user_id = "cli_user"
        self._chat_id = "cli_chat"

    async def start(self):
        print("\n--- Galatea CLI Mode ---")
        print("Type your message and press Enter. Type 'exit' to quit.\n")
        
        # Start listening for outbound messages in a background task
        asyncio.create_task(self._listen_outbound())
        
        # Start the CLI input loop
        while True:
            # We use aioconsole or a simple loop for input
            # To avoid blocking the event loop, we use run_in_executor
            user_input = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            user_input = user_input.strip()
            
            if user_input.lower() in ("exit", "quit"):
                print("Exiting CLI Mode...")
                sys.exit(0)
                
            if user_input:
                inbound = InboundMessage(
                    source="cli",
                    user_id=self._user_id,
                    chat_id=self._chat_id,
                    text=user_input
                )
                await self._bus.publish_inbound(inbound)

    async def _listen_outbound(self):
        async for msg in self._bus.subscribe_outbound("cli"):
            print(f"\n[Galatea]: {msg.text}\n")
            print("> ", end="", flush=True)
