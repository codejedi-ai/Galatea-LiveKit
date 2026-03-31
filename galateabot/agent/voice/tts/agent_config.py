import json
import os
from text_utils import ArcanaSentenceTokenizer

def load_voice_configs():
    config_path = os.path.join(os.path.dirname(__file__), "agentconfig.json")
    
    if not os.path.exists(config_path):
        # Fallback or empty dict if file doesn't exist
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    # Post-process configs to instantiate objects
    for voice_key, config in configs.items():
        if "tokenizer_config" in config:
            tok_conf = config["tokenizer_config"]
            if tok_conf.get("type") == "ArcanaSentenceTokenizer":
                min_len = tok_conf.get("min_sentence_len", 1000)
                config["sentence_tokenizer"] = ArcanaSentenceTokenizer(min_sentence_len=min_len)
            # Remove the raw config entry if desired, or keep it. 
            # Keeping it doesn't hurt, but "sentence_tokenizer" key is what the agent expects.

    return configs

VOICE_CONFIGS = load_voice_configs()
