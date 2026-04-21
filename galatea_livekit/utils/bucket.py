import json
from typing import Any, List, Optional
from galatea_livekit.utils.paths import PathManager

class BucketStore:
    def __init__(self, bucket_name: str = "default"):
        self.bucket_dir = PathManager.get_bucket_dir(bucket_name)

    def put(self, key: str, data: Any):
        """Put a JSON object into the bucket."""
        # Ensure key is a safe filename
        safe_key = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in key])
        filepath = self.bucket_dir / f"{safe_key}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(filepath)

    def get(self, key: str) -> Optional[Any]:
        """Get a JSON object from the bucket."""
        safe_key = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in key])
        filepath = self.bucket_dir / f"{safe_key}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_keys(self) -> List[str]:
        """List all keys (filenames without .json) in the bucket."""
        return [f.stem for f in self.bucket_dir.glob("*.json")]

    def delete(self, key: str):
        """Delete an object from the bucket."""
        safe_key = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in key])
        filepath = self.bucket_dir / f"{safe_key}.json"
        if filepath.exists():
            filepath.unlink()
