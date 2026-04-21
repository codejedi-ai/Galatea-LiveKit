import uuid
from livekit.agents import function_tool
from galatea_livekit.session.history import HistoryManager
from galatea_livekit.bus.events import HistoryRequest

@function_tool
def query_local_history(channel: str, chat_id: str, limit: int = 50) -> str:
    """Query the local database for message history of a specific chat."""
    import json
    try:
        mgr = HistoryManager()
        history = mgr.get_history(channel, chat_id, limit)
        if not history:
            return f"No local history found for {channel}:{chat_id}"
        return json.dumps(history)
    except Exception as e:
        return f"Error querying local history: {e}"

@function_tool
def request_channel_history(channel: str, chat_id: str, limit: int = 100) -> str:
    """Request a channel (e.g. 'telegram') to scan and return its history for a chat."""
    from main import GLOBAL_BUS
    if not GLOBAL_BUS:
        return "Error: Message bus not available."
    
    async def _request():
        req = HistoryRequest(
            channel=channel,
            chat_id=chat_id,
            limit=limit,
            request_id=str(uuid.uuid4())
        )
        await GLOBAL_BUS.publish_history_request(req)
        
    import asyncio
    asyncio.create_task(_request())
    return f"History request sent to {channel} for {chat_id}. Response will be sent to the message bus."
