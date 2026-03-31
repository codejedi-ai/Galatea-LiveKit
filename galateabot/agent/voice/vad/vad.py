from livekit.plugins import silero

def load_vad():
    return silero.VAD.load()
