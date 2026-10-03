"""
Memory and disk cache for query results.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class QueryCache:
    """
    Stores query results in memory and also saves them to JSON files
    to be called if the program is rerun. If anything goes wrong, a
    warning is printed and the file is skipped.

    Attributes:
        - cache_dir: Directory to save cache files.
        - memory_cache: Results stored in the current run.
        - disk_enabled: False if folder creation fails, so results
                        would only be in memory until the program ends.
    """
    def __init__(self, cache_dir: str = "data/cache") -> None:
        """
        Sets up the cache and creates its folder if needed.

        Args:
            - cache_dir: Directory to save cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.memory_cache: Dict[str, Any] = {}
        self.disk_enabled: bool = True
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.disk_enabled = False
            print(f"Cache directory: {self.cache_dir} is unusable, "
                  "falling back to in-memory caching only.", file=sys.stderr)

    def _key_to_filename(self, key: str) -> str:
        """
        Turns a key into a safe file name.

        Args:
            - key: The cache key.

        Returns:
            A fixed length name without a file extension.
        """
        # hashlib.md5().hexdigest() converts unsafe input strings
        # into fixed length hexadecimal strings
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Gets a safe result. Checks memory first then saved cache files.
        Damaged files are deleted.

        Args:
            - key: The cache key.

        Returns:
            The saved result or None if there isn't one.
        """
        if key in self.memory_cache:
            return self.memory_cache[key]
        if not self.disk_enabled:
            return None

        cache_file = self.cache_dir / f"{self._key_to_filename(key)}.json"
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                result = json.load(f)
        except FileNotFoundError:
            return None
        except ValueError:
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        except OSError:
            print(f"Could not read cache file {cache_file}", file=sys.stderr)
            return None

        # Populates memory with disk result
        self.memory_cache[key] = result
        return result

    def set(self, key: str, value: Any) -> None:
        """
        Saves a result under a key. The result is always stored
        in memory. If saving to file fails, a warning is printed
        and the program continues.

        Args:
            - key: The cache key.
            - value: The result to save. Must be plain data.
        """
        # Saves value to memory
        self.memory_cache[key] = value
        if not self.disk_enabled:
            return

        cache_file = self.cache_dir / f"{self._key_to_filename(key)}.json"
        try:
            # Recreate the folder in case it was deleted
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(value, f)
        except (OSError, TypeError, ValueError) as e:
            print(f"Warning: could not write cache entry "
                  f"({type(e).__name__}: {e}). Continuing without it.",
                  file=sys.stderr)
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> None:
        """
        Deletes all saved cache files from memory and disk.
        If a file cannot be deleted, a warning is printed and the
        rest are still removed.
        """
        self.memory_cache.clear()
        if not self.disk_enabled:
            return
        try:
            files = list(self.cache_dir.glob("*.json"))
        except OSError as e:
            print(f"Warning: could not list cache directory ({e}).",
                  file=sys.stderr)
            return

        for f in files:
            try:
                f.unlink(missing_ok=True)
            except OSError as e:
                print(f"Could not delete cache file {f} ({e})",
                      file=sys.stderr)


# Shared instance used by retriever and indexer
query_cache = QueryCache()
