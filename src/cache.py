import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


class QueryCache:
    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}

    def _key_to_filename(self, key: str) -> str:

        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:

        if key in self.memory_cache:
            return self.memory_cache[key]

        filename = self._key_to_filename(key)
        cache_file = self.cache_dir / f"{filename}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                result = pickle.load(f)
                self.memory_cache[key] = result
                return result
        return None

    def set(self, key: str, value: Any) -> None:
        self.memory_cache[key] = value
        filename = self._key_to_filename(key)
        cache_file = self.cache_dir / f"{filename}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        self.memory_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()


query_cache = QueryCache()
