from typing import Any, List, Optional
from livekit.agents import function_tool
from galateabot.utils.bucket import BucketStore

@function_tool
def bucket_put(bucket: str, key: str, data: str) -> str:
    """Put an object into a specific bucket. Data should be a JSON-parseable string."""
    import json
    try:
        parsed_data = json.loads(data)
        store = BucketStore(bucket)
        path = store.put(key, parsed_data)
        return f"Stored {key} in bucket {bucket} at {path}"
    except Exception as e:
        return f"Error storing in bucket: {e}"

@function_tool
def bucket_get(bucket: str, key: str) -> str:
    """Retrieve an object from a bucket."""
    import json
    try:
        store = BucketStore(bucket)
        data = store.get(key)
        if data is None:
            return f"Key {key} not found in bucket {bucket}"
        return json.dumps(data)
    except Exception as e:
        return f"Error reading from bucket: {e}"

@function_tool
def bucket_list(bucket: str) -> str:
    """List all keys in a bucket."""
    try:
        store = BucketStore(bucket)
        keys = store.list_keys()
        return ", ".join(keys) if keys else "Bucket is empty"
    except Exception as e:
        return f"Error listing bucket: {e}"
