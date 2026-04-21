import os
from pathlib import Path

class PathManager:
    # GALATEA_MODE can be 'prod' or 'test'
    MODE = os.getenv("GALATEA_MODE", "test")
    
    @classmethod
    def get_root(cls) -> Path:
        # Priority: ENV > Default
        env_root = os.getenv("GALATEA_ROOT")
        if env_root:
            root = Path(env_root)
        elif cls.MODE == "prod":
            root = Path.home() / ".galatea"
        else:
            # Local testing path as default fallback
            root = Path("/Volumes/PHILIPS/1-repos/1-GalateaAI/galatea_livekit/.galatea")
        
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def get_config_dir(cls) -> Path:
        c = cls.get_root() / "config"
        c.mkdir(parents=True, exist_ok=True)
        return c

    @classmethod
    def get_data_dir(cls) -> Path:
        d = cls.get_root() / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def get_bucket_dir(cls, bucket_name: str = "default") -> Path:
        b = cls.get_data_dir() / "bucket" / bucket_name
        b.mkdir(parents=True, exist_ok=True)
        return b

    @classmethod
    def get_db_path(cls, db_name: str) -> str:
        return str(cls.get_data_dir() / f"{db_name}.db")

    @classmethod
    def get_config_path(cls) -> Path:
        return cls.get_root() / "config.json"
