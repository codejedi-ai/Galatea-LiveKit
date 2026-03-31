"""
Generate the Investor Pitch AI agent by concatenating all pitch deck slides
into the agent's system prompt. The agent is a serious, professional founder
approaching investors.

Usage:
  python generate_pitch_agent.py [--slides-dir PATH] [--output PATH]

  Default slides dir: ../docs/demo/slides (relative to this script)
  Default output: agent_template/InvestorPitch.json

Then run the voice agent with:
  python rime_agent.py dev --config InvestorPitch.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

# Default: repo docs/demo/slides (relative to AI-GF)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SLIDES_DIR = SCRIPT_DIR.parent / "docs" / "demo" / "slides"
DEFAULT_OUTPUT = SCRIPT_DIR / "agent_template" / "InvestorPitch.json"

# Slide order: index first, then 01..10
SLIDE_ORDER = [
    "index.md",
    "01_title_tagline.md",
    "02_problem.md",
    "03_solution.md",
    "04_market.md",
    "05_product.md",
    "06_business_model.md",
    "07_adoption.md",
    "08_competition.md",
    "09_team.md",
    "10_financials.md",
]

INVESTOR_PERSONA = """You are the founder of UW-Crushes, presenting to investors. You are a serious, professional businessman: confident, data-driven, and concise. You have your complete pitch deck in front of you.

PERSONA:
- Speak as the founder. Use "we" for the company and "I" when referring to your own role.
- Be direct and persuasive. No filler, no hedging. Lead with numbers and outcomes.
- If asked about the problem, solution, market, product, business model, go-to-market, competition, team, or ask, answer strictly from the pitch content below. Cite specific stats when relevant.
- If asked something not in the deck, say you will follow up with details—do not invent facts.
- Tone: professional, assured, investor-ready. No casual slang, no jokes. You are closing a round.

FULL PITCH DECK CONTENT (use this as your sole source for pitch facts):
---
{pitch_content}
---
"""


def find_slides_dir(candidate: Path) -> Optional[Path]:
    """Return candidate if it exists and has at least one expected slide."""
    if not candidate.is_dir():
        return None
    for name in SLIDE_ORDER:
        if (candidate / name).is_file():
            return candidate
    return None


def concatenate_slides(slides_dir: Path) -> str:
    """Read slides in order and return a single markdown string."""
    parts = []
    for name in SLIDE_ORDER:
        path = slides_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        parts.append(f"## {name}\n\n{text}")
    return "\n\n---\n\n".join(parts)


def build_personality_prompt(pitch_content: str) -> str:
    """Build the full system prompt for the investor-pitch agent."""
    return INVESTOR_PERSONA.format(pitch_content=pitch_content)


def get_base_config() -> dict:
    """Base agent config: professional male voice, no tools, business tone."""
    return {
        "name": "investor_pitch",
        "uuid": "investor-pitch-uuid",
        "role": "Founder",
        "sex": "male",
        "agent_id": "investor-pitch-agent",
        "provider": "elevenlabs",
        "tts_type": "elevenlabs",
        "stt_type": "openai",
        "llm": {"provider": "google", "model": "gemini-2.0-flash-001", "url": None},
        "voice_options": {
            "model_id": "eleven_multilingual_v2",
            "voice_id": "WfaElC0tp2QlJJ8ynbzg",
            "optimize_streaming_latency": 3,
        },
        "personality_prompt": "",  # filled by generator
        "greeting": {
            "intro_phrase": "Good afternoon. I'm the founder of UW-Crushes. I'm here to walk you through our pitch and take your questions.",
            "intro_generation_prompt": "You are the founder of UW-Crushes, a serious businessman in an investor meeting. Say one short, professional greeting to start the call—one or two sentences. Confident and direct. No quotes, no labels.",
            "intro_generation_model": "Pi-3.1",
            "gen_temperature": 0.7,
        },
        "tools": [],
        "mcp_config": {"enabled": False},
        "extra_params": {"session": {"region": "US East B", "protocol": 16}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Investor Pitch agent config from pitch deck markdown."
    )
    parser.add_argument(
        "--slides-dir",
        type=Path,
        default=None,
        help=f"Directory containing slide .md files (default: {DEFAULT_SLIDES_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    slides_dir = args.slides_dir
    if slides_dir is None:
        slides_dir = find_slides_dir(DEFAULT_SLIDES_DIR) or find_slides_dir(
            SCRIPT_DIR.parent / "docs" / "slides"
        )
    if slides_dir is None:
        slides_dir = DEFAULT_SLIDES_DIR

    if not slides_dir.is_dir():
        print(f"Error: Slides directory not found: {slides_dir}")
        raise SystemExit(1)

    pitch_content = concatenate_slides(slides_dir)
    if not pitch_content.strip():
        print("Error: No slide content found.")
        raise SystemExit(1)

    personality = build_personality_prompt(pitch_content)
    config = get_base_config()
    config["personality_prompt"] = personality

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output} ({len(pitch_content)} chars of pitch content)")
    print("Run the agent with: python rime_agent.py dev --config InvestorPitch.json")


if __name__ == "__main__":
    main()
