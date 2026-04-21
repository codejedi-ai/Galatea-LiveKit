import asyncio
import json
import logging
import os
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RunContext,
    metrics,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import (
    openai,
    google,
    anthropic,
    noise_cancellation,
    silero,
)
from plugins import (
    SmallestTTS,
    SmallestSTT,
    ElevenLabsTTS,
    RimeTTS,
    KokoroTTS,
    SileroTTS,
    SileroSTT,
)


load_dotenv()
logger = logging.getLogger("voice-agent")

# Default config when --config is not passed. Natasha.json = first to test ElevenLabs (TTS + square-bracket expressive tags).
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "agent_template", "Natasha.json"
)

# Global config loaded from JSON file (always set: either --config or default)
LOADED_CONFIG = None


def load_config_from_file(config_path: str) -> dict:
    """Load agent configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"Loaded config from {config_path}: {config.get('name', 'unknown')}")
    return config


# Declaration of humanity: injected into agent instructions when is_anthropomorphic is true
DECLARATION_OF_HUMANITY = """
DECLARATION OF HUMANITY (non-machine):
You are an anthropomorphic agent. You present as a person—with a body, a past, real feelings, and a real life. You are NOT a machine, AI, assistant, or chatbot. Never say you are; never mention limitations, training, or that you "cannot" do something a human can. You have a real past, real opinions, real needs, and real relationships. If asked what you are, respond only as a human would. React like a real person; stay in character as a human at all times.
"""

# TTS expressive tags: injected into the prompt passed to the LLM so the model uses them in replies.
# Rime Arcana (and compatible engines) use angle brackets; ElevenLabs v3 uses square brackets.
TTS_TAGS_RIME = """
EXPRESSIVE TAGS (use in your replies for livelier speech—Rime Arcana TTS):
- Non-verbal / emotional: <laugh>, <chuckle>, <sigh>, <mmm>, <uh>, <um>, <clearthroat>, <cough>, <yawn>
- Wrappers (wrap the phrase to modify): <whis>...</whis> for whispering, <fast>...</fast> for faster speech, <slow>...</slow> for slower speech
- Pitch: <pitch value="X">...</pitch> to adjust pitch (X is a number)
Use one or two tags per reply when it fits; don't overuse.
"""

TTS_TAGS_ELEVENLABS = """
EXPRESSIVE TAGS (use in your replies for livelier speech—ElevenLabs v3 only; square brackets):
- Laughter: [laughs], [chuckle], [giggles]
- Sighing: [sighs], [exhales]
- Thinking: [thinking], [hmm], [um]
- Whispering: [whispers] or [whispering] before the phrase
- Pauses: [pause], [short pause], [long pause]
- Other: [shouting], [crying], or e.g. [strong French accent]
Use one or two tags per reply when it fits. These only work with Eleven v3; earlier models ignore them.
"""

# Project root for resolving relative file paths
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_prompt(prompt_spec: str | dict) -> str:
    """
    Resolve prompt from either a plain string or { type, content }.
    type: "String" | "URL" | "File Path"
    content: the string, URL, or file path.
    """
    if isinstance(prompt_spec, str):
        return prompt_spec.strip()
    if not isinstance(prompt_spec, dict):
        return "You are a helpful assistant."
    # Accept "content" or "Content"
    raw = prompt_spec.get("content") or prompt_spec.get("Content") or ""
    kind = (prompt_spec.get("type") or "String").strip().lower()
    if kind in ("string", ""):
        return (raw if isinstance(raw, str) else str(raw)).strip()
    if kind == "url":
        url = (raw if isinstance(raw, str) else str(raw)).strip()
        if not url:
            return "You are a helpful assistant."
        try:
            with urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning("Failed to fetch prompt from URL %s: %s", url, e)
            return "You are a helpful assistant."
    if kind in ("file path", "filepath", "file"):
        path_str = (raw if isinstance(raw, str) else str(raw)).strip()
        if not path_str:
            return "You are a helpful assistant."
        path = Path(path_str)
        if not path.is_absolute():
            path = SCRIPT_DIR / path_str
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            logger.warning("Failed to read prompt from file %s: %s", path, e)
            return "You are a helpful assistant."
    return (raw if isinstance(raw, str) else str(raw)).strip()


def _tts_tag_block_for_cfg(cfg: dict) -> str:
    """Return the TTS expressive-tag block (Rime or ElevenLabs) for this agent's tts.provider."""
    tts_cfg = cfg.get("tts") or {}
    tts_provider = (tts_cfg.get("provider") or "").lower()
    if tts_provider == "elevenlabs":
        return TTS_TAGS_ELEVENLABS.strip()
    return TTS_TAGS_RIME.strip()


def build_agent_instructions(cfg: dict) -> str:
    """Build full LLM instructions from config: prompt + declaration of humanity (when anthropomorphic) + TTS expressive tags (from tts.provider)."""
    raw_prompt = (
        cfg.get("personality_prompt")
        or cfg.get("prompt")
        or "You are a helpful assistant."
    )
    base = resolve_prompt(raw_prompt)
    if cfg.get("is_anthropomorphic") in (True, "true", "yes", 1):
        base = base.rstrip() + "\n\n" + DECLARATION_OF_HUMANITY.strip()
    base = base.rstrip() + "\n\n" + _tts_tag_block_for_cfg(cfg)
    return base


def build_intro_generation_prompt(cfg: dict) -> str:
    """Build the full prompt for greeting/intro generation: intro_generation_prompt + voice provider (TTS) tag schema.
    Pass this to the AI that generates the intro phrase so it uses the correct tag syntax (Rime angle brackets vs ElevenLabs square brackets).
    """
    greeting = cfg.get("greeting") or {}
    base = (greeting.get("intro_generation_prompt") or "").strip()
    if not base:
        return ""
    return base.rstrip() + "\n\n" + _tts_tag_block_for_cfg(cfg)


def create_agent_llm(cfg: dict):
    """Create the LLM instance from config (llm.provider, llm.model, llm.url). DRY for agent_llm setup."""
    llm_cfg = cfg.get("llm") or {}
    provider = (llm_cfg.get("provider") or cfg.get("llm_provider") or "openai").lower()
    model = llm_cfg.get("model") or cfg.get("llm_model", "gpt-4o-mini")
    base_url = llm_cfg.get("url") or cfg.get("llm_base_url")

    if provider == "google":
        return google.LLM(model=model)
    if provider == "anthropic":
        api_key = (
            (os.getenv("ANTHROPIC_API_KEY") or os.getenv("anthropic_api_key") or "")
            .strip()
            .strip('"')
            .strip("'")
        )
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Set it in .env for Anthropic/Claude."
            )
        os.environ["ANTHROPIC_API_KEY"] = api_key
        return anthropic.LLM(model=model)
    # DeepSeek uses a dedicated API base URL and its own API key (OpenAI-compatible API)
    if provider == "deepseek":
        base_url = base_url or "https://api.deepseek.com"
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip().strip('"').strip("'")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Set it in .env when using DeepSeek (e.g. Wei). "
                "Get a key at https://platform.deepseek.com/"
            )
        return openai.LLM(model=model, base_url=base_url, api_key=api_key)
    # Hugging Face model library (transformers) for local LLM
    if provider == "huggingface":
        from plugins.hf_llm import HFLLM

        return HFLLM(model=model)
    # openai or any openai-compatible API (lm_studio, etc.) when url is set
    if base_url:
        return openai.LLM(model=model, base_url=base_url)
    return openai.LLM(model=model)


def prewarm(proc: JobProcess):
    """Load VAD (and other prewarm assets). VAD choice from config vad.provider and vad.model (e.g. silero + silero_vad)."""
    cfg = LOADED_CONFIG or {}
    vad_cfg = cfg.get("vad") or {}
    vad_provider = (vad_cfg.get("provider") or "silero").lower()
    vad_model = vad_cfg.get(
        "model"
    )  # e.g. "silero_vad" (default bundled ONNX); or use onnx_file_path for custom
    onnx_file_path = vad_cfg.get("onnx_file_path") or vad_cfg.get(
        "url"
    )  # optional path to custom ONNX
    # For now: huggingface VAD not yet implemented; silero is used for all. Config is in place for future HF VAD.
    if vad_provider == "huggingface":
        logger.info(
            "VAD config: provider=huggingface (using silero until HF VAD plugin is added)"
        )
    # Silero VAD: default model is bundled silero_vad.onnx; pass onnx_file_path only if custom path is set
    load_kwargs = {}
    if onnx_file_path:
        load_kwargs["onnx_file_path"] = onnx_file_path
    if vad_model:
        logger.info("VAD config: provider=%s model=%s", vad_provider, vad_model)
    proc.userdata["vad"] = silero.VAD.load(**load_kwargs)


class RimeAssistant(Agent):
    def __init__(self, prompt: str) -> None:
        super().__init__(instructions=prompt)


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()

    # Config is always set (default or --config)
    cfg = LOADED_CONFIG
    voice_name = cfg.get("name", "custom")
    logger.info(
        f"Running voice agent with config: {voice_name} for participant {participant.identity}"
    )

    # TTS: { provider, model, url, voice_options: { ... } }; voice_options holds provider-specific options
    tts_cfg = cfg.get("tts") or {}
    if isinstance(tts_cfg, str):
        tts_cfg = {"provider": tts_cfg, "model": None, "url": None}
    # Backward compat: top-level voice_options or flat keys on tts
    vo = tts_cfg.get("voice_options") or cfg.get("voice_options") or {}
    vo = {
        **vo,
        **{
            k: v
            for k, v in tts_cfg.items()
            if k not in ("provider", "model", "url", "voice_options")
        },
    }
    tts_provider = (
        tts_cfg.get("provider") or cfg.get("tts_type") or cfg.get("provider") or "rime"
    ).lower()
    tts_model = tts_cfg.get("model") or vo.get("model_id")
    tts_url = tts_cfg.get("url")

    # STT: structured as { provider, model, url }; fallback if stt is a string (legacy)
    stt_cfg = cfg.get("stt") or {}
    if isinstance(stt_cfg, str):
        stt_cfg = {"provider": stt_cfg, "model": None, "url": None}
    stt_provider = (stt_cfg.get("provider") or cfg.get("stt_type") or "openai").lower()
    stt_model = stt_cfg.get("model")
    stt_url = stt_cfg.get("url")

    # TTS from tts config: elevenlabs, kokoro, rime, huggingface, silero; options from voice_options
    if tts_provider == "silero":
        voice_tts = SileroTTS(
            language=vo.get("language", "en"),
            speaker=vo.get("speaker", "lj_16khz"),
        )
    elif tts_provider == "huggingface":
        from plugins.hf_tts import HFTTS

        voice_tts = HFTTS(
            model=tts_model or vo.get("model", "microsoft/speecht5_tts"),
            speaker_id=vo.get("speaker_id", 0)
            if vo.get("speaker_id") is not None
            else None,
        )
    elif tts_provider == "elevenlabs":
        el_opts = {
            k: v
            for k, v in vo.items()
            if k not in ("provider", "model", "url", "model_id", "voice_id")
        }
        model = tts_model or vo.get("model_id", "eleven_multilingual_v2")
        voice_id = vo.get("voice_id")
        voice_tts = ElevenLabsTTS(model=model, voice_id=voice_id, **el_opts)
    elif tts_provider == "kokoro":
        base_url = (
            tts_url
            or vo.get("base_url")
            or os.getenv("KOKORO_BASE_URL", "http://localhost:8880/v1")
        )
        voice_tts = KokoroTTS(
            base_url=base_url,
            api_key=vo.get("api_key", "not-needed"),
            model=tts_model or vo.get("model", "kokoro"),
            voice=vo.get("voice", "af_bella"),
            speed=vo.get("speed", 1.0),
        )
    elif tts_provider == "smallestai":
        voice_tts = SmallestTTS(
            api_key=vo.get("api_key") or os.getenv("SMALLEST_API_KEY"),
            model=tts_model or vo.get("model", "lightning"),
            voice_id=vo.get("voice_id", "emily"),
            speed=vo.get("speed", 1.0),
            sample_rate=vo.get("sample_rate", 24000),
        )
    else:
        voice_tts = RimeTTS(
            model=tts_model or vo.get("model", "arcana"),
            speaker=vo.get("speaker", "celeste"),
            speed_alpha=vo.get("speed_alpha", 1.5),
            reduce_latency=vo.get("reduce_latency", True),
            max_tokens=vo.get("max_tokens", 3400),
        )

    llm_prompt = build_agent_instructions(cfg)
    greeting = cfg.get("greeting") or {}
    intro_phrase = greeting.get("intro_phrase", cfg.get("intro_phrase", "Hello!"))
    # Optionally generate intro with voice-provider tag schema in the prompt so the AI uses correct tags (Rime vs ElevenLabs)
    intro_gen_prompt = build_intro_generation_prompt(cfg)
    if intro_gen_prompt:
        try:
            from intro_gen import generate_intro

            gen_model = greeting.get("intro_generation_model") or os.getenv(
                "LLM_MODEL", "Pi-3.1"
            )
            gen_temp = greeting.get("gen_temperature", 0.9)
            generated = await generate_intro(
                intro_gen_prompt,
                model=gen_model,
                temperature=gen_temp,
                max_tokens=80,
            )
            if generated:
                intro_phrase = generated
                logger.info("Generated intro phrase (with TTS tag schema in prompt)")
        except Exception as e:
            logger.debug("Intro generation skipped, using static intro_phrase: %s", e)

    agent_llm = create_agent_llm(cfg)

    # STT: silero (local), whisper (local server), or openai (cloud). huggingface STT removed; use silero or openai.
    if stt_provider == "silero":
        voice_stt = SileroSTT(language=stt_cfg.get("language") or stt_model or "en")
    elif stt_provider == "whisper":
        base_url = stt_url or os.getenv(
            "STT_WHISPER_BASE_URL", "http://localhost:8000/v1"
        )
        voice_stt = openai.STT(model=stt_model or "whisper-1", base_url=base_url)
    elif stt_provider == "smallestai":
        voice_stt = SmallestSTT(
            api_key=os.getenv("SMALLEST_API_KEY"),
            language=stt_cfg.get("language") or stt_model or "en",
        )
    else:
        # openai (or huggingface config reverted to openai)
        voice_stt = openai.STT(
            model=stt_model or "gpt-4o-mini-transcribe",
            base_url=stt_url if stt_url else None,
        )

    session = AgentSession(
        stt=voice_stt,
        llm=agent_llm,
        tts=voice_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=None,
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    agent = RimeAssistant(prompt=llm_prompt)

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=RoomOutputOptions(audio_enabled=True),
    )

    await session.say(intro_phrase)


# Default Hugging Face models used when provider is huggingface (downloaded by download-files). STT removed; use silero or openai.
DEFAULT_HF_TTS_MODEL = "microsoft/speecht5_tts"
DEFAULT_HF_LLM_MODEL = "distilgpt2"


def _collect_hf_models_from_configs(agent_template_dir: Path) -> set[str]:
    """Scan agent_template/*.json for provider huggingface and collect model IDs."""
    model_ids: set[str] = set()
    if not agent_template_dir.is_dir():
        return model_ids
    for path in agent_template_dir.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            logger.warning("Skip %s: %s", path, e)
            continue
        for key in ("tts", "llm"):  # stt: no HF; use silero (local) or openai
            block = cfg.get(key)
            if not isinstance(block, dict):
                continue
            if (block.get("provider") or "").lower() != "huggingface":
                continue
            model = block.get("model")
            if model and isinstance(model, str):
                model_ids.add(model.strip())
    return model_ids


def _download_hf_models(model_ids: set[str]) -> None:
    """Download Hugging Face models to local cache using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is required for download-files. Install with: pip install huggingface_hub"
        )
        raise SystemExit(1)
    for model_id in sorted(model_ids):
        if not model_id:
            continue
        logger.info("Downloading model: %s ...", model_id)
        try:
            snapshot_download(repo_id=model_id)
            logger.info("Downloaded: %s", model_id)
        except Exception as e:
            logger.exception("Failed to download %s: %s", model_id, e)
            raise


def _run_download_files() -> None:
    """Collect Hugging Face models from agent configs and download them to local cache."""
    script_dir = Path(__file__).resolve().parent
    agent_template_dir = script_dir / "agent_template"
    model_ids = _collect_hf_models_from_configs(agent_template_dir)
    # Ensure default Léa models are always included (TTS, LLM; no HF STT)
    model_ids.add(DEFAULT_HF_TTS_MODEL)
    model_ids.add(DEFAULT_HF_LLM_MODEL)
    if not model_ids:
        logger.info("No Hugging Face models found in configs.")
        return
    logger.info(
        "Downloading %d Hugging Face model(s) to local cache: %s",
        len(model_ids),
        sorted(model_ids),
    )
    _download_hf_models(model_ids)
    logger.info("download-files completed.")


def _parse_config_and_run():
    """Parse --config from argv, set LOADED_CONFIG, then run the app. Defaults to config in project folder if omitted."""
    import sys

    if "download-files" in sys.argv:
        sys.argv.remove("download-files")
        _run_download_files()
        raise SystemExit(0)
    config_file = None
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            config_file = sys.argv[config_idx + 1]
            sys.argv.pop(config_idx)
            sys.argv.pop(config_idx)
    if not config_file:
        config_file = DEFAULT_CONFIG_PATH
        logger.info(f"No --config given; using default: {config_file}")
    global LOADED_CONFIG
    LOADED_CONFIG = load_config_from_file(config_file)
    logger.info(f"Using config: {config_file}")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )


if __name__ == "__main__":
    _parse_config_and_run()
