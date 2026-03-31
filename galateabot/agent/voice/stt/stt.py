from livekit.plugins import google

def get_stt(provider="google", **kwargs):
    if provider == "google":
        # Gemini RealtimeModel handles STT natively if not in half-cascade
        return None 
    return None
