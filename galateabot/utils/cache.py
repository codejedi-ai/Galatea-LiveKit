import json
import logging
import os
from typing import Any, Optional
try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger("galatea-cache")

class Cache:
    def __init__(self, host="localhost", port=6379, db=0):
        self._redis = None
        if redis:
            try:
                self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                # Test connection
                self._redis.ping()
                logger.info("Connected to Redis cache.")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}. Falling back to in-memory cache.")
                self._redis = None
        else:
            logger.warning("Redis package not installed. Falling back to in-memory cache.")
        
        self._local_cache = {}

    def set(self, key: str, value: Any, expire: int = None):
        if self._redis:
            self._redis.set(key, json.dumps(value), ex=expire)
        else:
            self._local_cache[key] = value

    def get(self, key: str) -> Optional[Any]:
        if self._redis:
            val = self._redis.get(key)
            return json.loads(val) if val else None
        else:
            return self._local_cache.get(key)

    def delete(self, key: str):
        if self._redis:
            self._redis.delete(key)
        else:
            if key in self._local_cache:
                del self._local_cache[key]

# Singleton instance
cache = Cache()
