from livekit.plugins import google

def get_tts(provider="google", **kwargs):
    if provider == "google":
        # Gemini RealtimeModel handles TTS natively 
        return None
    return None
