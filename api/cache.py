"""Simple thread-safe in-memory response cache.

Caches raw, format-agnostic data so repeated requests (Shiny hot path, agent
tool calls) read from memory instead of re-reading Parquet from disk. TTL-based
expiry — data updates only once per month, so no explicit invalidation needed
(though POST /cache/clear exists for the monthly deploy / testing).
"""

import hashlib
import time
from threading import Lock


class ResponseCache:
    def __init__(self, default_ttl_seconds: int = 3600):
        self._cache: dict = {}
        self._lock = Lock()
        self.default_ttl = default_ttl_seconds

    def make_key(self, endpoint: str, params: dict) -> str:
        """Stable key from endpoint + params. 'format' is excluded — raw data
        is cached and the format is generated on read."""
        sorted_params = sorted(
            {k: v for k, v in params.items() if k != "format"}.items()
        )
        key_string = f"{endpoint}:{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str):
        """Return cached value, or None if missing/expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                return None
            return entry["data"]

    def set(self, key: str, data, ttl_seconds: int | None = None):
        """Store data with an expiry."""
        ttl = ttl_seconds or self.default_ttl
        with self._lock:
            self._cache[key] = {
                "data": data,
                "expires_at": time.time() + ttl,
                "created_at": time.time(),
            }

    def clear(self):
        """Clear all entries. Called after the pipeline deploys new data."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        """Cache statistics for the /health endpoint."""
        with self._lock:
            now = time.time()
            active = sum(1 for e in self._cache.values() if now <= e["expires_at"])
            return {
                "total_entries": len(self._cache),
                "active_entries": active,
                "expired_entries": len(self._cache) - active,
            }


# Singleton — one cache shared across all requests.
response_cache = ResponseCache(default_ttl_seconds=3600)  # 1 hour default TTL
