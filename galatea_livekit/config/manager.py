import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from galatea_livekit.utils.paths import PathManager

logger = logging.getLogger("config-manager")

class ConfigManager:
    _configs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def load(cls):
        """Load all config files from .galatea/config/"""
        config_dir = PathManager.get_config_dir()
        for config_file in config_dir.glob("*.json"):
            name = config_file.stem
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cls._configs[name] = json.load(f)
                logger.info(f"Loaded config: {name}")
            except Exception as e:
                logger.error(f"Failed to load config {name}: {e}")

    @classmethod
    def get(cls, config_name: str, key_path: str = None, default: Any = None) -> Any:
        """
        Get value from a specific config file.
        Example: get("llm", "text.model") -> looks in llm.json for text -> model
        """
        config = cls._configs.get(config_name)
        if config is None:
            return default
        
        if key_path is None:
            return config

        keys = key_path.split(".")
        val = config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

# Pre-load on import
ConfigManager.load()
