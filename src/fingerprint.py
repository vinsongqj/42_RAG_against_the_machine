from pathlib import Path
from typing import Any, Callable, Optional


def dir_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        return str(int(path.stat().st_mtime))
    latest = 0.0
    for p in path.rglob("*"):
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
    return str(int(latest))


class FingerprintedCache:

    def __init__(self, loader: Callable[[str], Any]) -> None:
        self._loader = loader
        self._key: Optional[str] = None
        self._obj: Optional[Any] = None
        self._fingerprint: Optional[str] = None

    def get(self, key: str, fingerprint_path: Path) -> Any:
        current_fingerprint = dir_fingerprint(fingerprint_path)
        if self._obj is not None and self._key == key and current_fingerprint == self._fingerprint:
            return self._obj

        self._obj = self._loader(key)
        self._key = key
        self._fingerprint = current_fingerprint
        return self._obj

    def set(self, key: str, obj: Any, fingerprint_path: Path) -> None:
        self._key = key
        self._obj = obj
        self._fingerprint = dir_fingerprint(fingerprint_path)

    def invalidate(self) -> None:
        self._key = self._obj = self._fingerprint = None
