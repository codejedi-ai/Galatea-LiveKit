import logging
from livekit.agents import function_tool
from galateabot.bus.events import OutboundMessage

logger = logging.getLogger("message-tools")

@function_tool
def send_message(target: str, user_id: str, chat_id: str, text: str) -> str:
    """
    Send a message to a specific channel/target.
    This bypasses the primary reasoning loop and directly queues a response.
    """
    from main import GLOBAL_BUS # We'll need access to the bus
    if not GLOBAL_BUS:
        return "Error: Message bus not available."
    
    async def _send():
        outbound = OutboundMessage(
            target=target,
            user_id=user_id,
            chat_id=chat_id,
            text=text
        )
        await GLOBAL_BUS.publish_outbound(outbound)
    
    import asyncio
    asyncio.create_task(_send())
    return f"Message queued for {target}:{chat_id}"
