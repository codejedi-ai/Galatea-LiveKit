import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key: return "MISSING"
    try:
        r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        return "VALID" if r.status_code == 200 else f"FAILED ({r.status_code})"
    except Exception as e: return f"ERROR ({e})"

def test_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: return "MISSING"
    try:
        # Anthropic doesn't have a simple GET user info, we'll try to list models if supported or just check auth
        r = requests.get("https://api.anthropic.com/v1/models", headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01"
        }, timeout=10)
        return "VALID" if r.status_code == 200 else f"FAILED ({r.status_code})"
    except Exception as e: return f"ERROR ({e})"

def test_elevenlabs():
    key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    if not key: return "MISSING"
    try:
        r = requests.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": key}, timeout=10)
        return "VALID" if r.status_code == 200 else f"FAILED ({r.status_code})"
    except Exception as e: return f"ERROR ({e})"

def test_deepseek():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: return "MISSING"
    try:
        r = requests.get("https://api.deepseek.com/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        return "VALID" if r.status_code == 200 else f"FAILED ({r.status_code})"
    except Exception as e: return f"ERROR ({e})"

def test_google():
    key = os.getenv("GOOGLE_API_KEY")
    if not key: return "MISSING"
    try:
        r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=10)
        return "VALID" if r.status_code == 200 else f"FAILED ({r.status_code})"
    except Exception as e: return f"ERROR ({e})"

def test_hf():
    key = os.getenv("HF_TOKEN")
    if not key: return "MISSING"
    try:
        r = requests.get("https://huggingface.co/api/whoami-v2", headers={"Authorization": f"Bearer {key}"}, timeout=10)
        return "VALID" if r.status_code == 200 else f"FAILED ({r.status_code})"
    except Exception as e: return f"ERROR ({e})"

if __name__ == "__main__":
    print("-" * 30)
    print(f"{'Service':<15} | {'Status':<15}")
    print("-" * 30)
    print(f"{'OpenAI':<15} | {test_openai()}")
    print(f"{'Anthropic':<15} | {test_anthropic()}")
    print(f"{'ElevenLabs':<15} | {test_elevenlabs()}")
    print(f"{'DeepSeek':<15} | {test_deepseek()}")
    print(f"{'Google':<15} | {test_google()}")
    print(f"{'HuggingFace':<15} | {test_hf()}")
    print("-" * 30)
