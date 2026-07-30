import hashlib
from pathlib import Path
from typing import Any, Optional
import pickle


class QueryCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}

    def _key_to_filename(self, key: str) -> str:
        """Convert a key to a safe filename using MD5 hash."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        # Check memory first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk
        filename = self._key_to_filename(key)
        cache_file = self.cache_dir / f"{filename}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
                self.memory_cache[key] = result
                return result
        return None

    def set(self, key: str, value: Any):
        self.memory_cache[key] = value
        filename = self._key_to_filename(key)
        cache_file = self.cache_dir / f"{filename}.pkl"
        # Ensure parent directory exists (it's the same cache_dir, but safe)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(value, f)

    def clear(self):
        self.memory_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()


# Global cache instance
query_cache = QueryCache()