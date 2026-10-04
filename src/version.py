"""
Detects when index files on disk have been changed so loaded data
can be reloaded only when necessary.
"""

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# Seconds a version result is reused before the folder is rechecked
DEFAULT_VERSION_LIFESPAN_SECS = 1.0


def dir_version(path: Path) -> str:
    """
    Returns a version string for a file or folder that changes
    whenever something in it is modified. A folder version is
    the most recent modified time of any file inside it. Files
    that disappear during the check are skipped.

    Args:
        - path: The file or folder to check:

    Returns:
        The version string. "missing" if the path doesn't exist or
        "unreadable" if it cannot be checked. Never raises.
    """
    try:
        if not path.exists():
            return "missing"
        if path.is_file():
            return str(int(path.stat().st_mtime))
    except OSError:
        return "unreadable"

    # For folders
    latest = 0.0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    # latest keeps track of the latest modification time
                    latest = max(latest, p.stat().st_mtime)
            except OSError:
                # If file disappears mid scan
                continue
    except OSError:
        # If folder fails
        pass
    return str(int(latest))


_version_lifespan_cache: Dict[str, Tuple[float, str]] = {}


def cached_dir_version(
        path: Path,
        lifespan: float = DEFAULT_VERSION_LIFESPAN_SECS) -> str:
    """
    Same as dir_version but remembers the answer briefly.
    If the answer was found within the version lifespan, the
    answer is returned without looking again. A change to the files
    may go unnoticed during the lifespan duration.

    Args:
        - path: The file or folder to check.
        - lifespan: How many seconds an answer is remembered.

    Returns:
        The version string.
    """
    key = str(path)
    now = time.monotonic()  # monotonic() is a clock that only moves forward
    cached = _version_lifespan_cache.get(key)
    # If current time - cached timestamp < lifespan, returns cached answer
    if cached is not None and now - cached[0] < lifespan:
        return cached[1]
    # If no cache or cache is too old
    version = dir_version(path)
    _version_lifespan_cache[key] = (now, version)
    return version


def refresh_dir_version(path: Path) -> str:
    """
    Checks current version, ignoring saved results, and saves it.
    Used after loading or building an index so the program doesn't
    mistake a new index for an outdated one.

    Args:
        - path: The file or folder to check.

    Returns:
        The current version string.
    """
    version = dir_version(path)
    _version_lifespan_cache[str(path)] = (time.monotonic(), version)
    return version


class VersionCache:
    """
    Keeps one loaded object in memory and reloads it when its files change.
    Used for the BM25 and semantic indexes. Loading them is slow so they are
    loaded once and reused. If the index is rebuilt, the next call notices and
    loads the new one. If reloading fails and an earlier copy exists in memory,
    it falls back to the earlier copy and prints a warning.
    """
    def __init__(self, loader: Callable[[str], Any]) -> None:
        """
        Args:
            - loader: Function that takes a key and returns the loaded object.
        """
        self._loader = loader
        self._key: Optional[str] = None
        self._obj: Optional[Any] = None
        self._version: Optional[str] = None

    def get(self, key: str, version_path: Path) -> Any:
        """
        Returns the loaded object, loading it first if needed.
        Loaded again if nothing is loaded yet, the key is different,
        or the files at version_path have changed since the last load.

        Args:
            - key: What to load, passed to the loader.
            - version_path: The file or folder to watch for changes.

        Returns:
            The loaded object.

        Raises:
            Whatever the loader raises if loading fails and no earlier copy
            exists to fall back on.
        """
        current_version = cached_dir_version(version_path)
        # If version hasn't changed, straight up return the stored object
        if (self._obj is not None and self._key == key
                and current_version == self._version):
            return self._obj
        # If version has changed, reload object
        try:
            obj = self._loader(key)
        except Exception as e:
            if self._obj is not None and self._key == key:
                print(f"Reloading {key!r} failed ({type(e).__name__}: {e}); "
                      "continuing with the previously loaded copy.",
                      file=sys.stderr)
                self._version = current_version
                return self._obj
            raise

        self._obj = obj
        self._key = key
        self._version = refresh_dir_version(version_path)

        return self._obj

    def set(self, key: str, obj: Any, version_path: Path) -> None:
        """
        Stores an object that was just built so it isn't loaded again.

        Args:
            - key: What the object was built for.
            - obj: The object to keep.
            - version_path: The file or folder to watch for changes.
        """
        self._key = key
        self._obj = obj
        self._version = refresh_dir_version(version_path)

    def invalidate(self) -> None:
        """
        Forgets the stored object so the next get() loads it again.
        """
        self._key = self._obj = self._version = None
