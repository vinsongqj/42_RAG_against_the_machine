import bm25s
from pathlib import Path
from typing import Any, List, Optional, cast
from src.models import MinimalSource
from src.cache import query_cache
from src.chunker import code_friendly_text

_retriever_cache: Optional[Any] = None
_index_dir_cache: Optional[str] = None


def _get_retriever(index_dir: str = "data/processed") -> Any:
    global _retriever_cache, _index_dir_cache
    if _retriever_cache is None or _index_dir_cache != index_dir:
        index_path = Path(index_dir) / "bm25_index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        print(f"Loading index from {index_path}...")
        _retriever_cache = bm25s.BM25.load(str(index_path), load_corpus=True)
        _index_dir_cache = index_dir
        print("Index loaded successfully!")
    return _retriever_cache


def retrieve(query: str, k: int = 5, index_dir: str = "data/processed",
             use_cache: bool = True) -> List[MinimalSource]:
    """Retrieve the top-k sources for a query.

    ``use_cache`` is an explicit parameter rather than something callers
    toggle by monkeypatching ``query_cache.get`` / ``set`` in place:
    that mutated shared global state (unsafe under a thread pool) and
    reassigned attributes to incompatible types.
    """
    # Cache key uses the *original* query so lookups are stable across the
    # tokenization transform below.
    cache_key = f"{query}::{k}::{index_dir}"
    if use_cache:
        cached = query_cache.get(cache_key)
        if cached is not None:
            return cast(List[MinimalSource], cached)

    retriever = _get_retriever(index_dir)

    # Apply the same identifier-expansion transform that was used at index
    # time — otherwise camelCase / snake_case queries cannot match.
    query_tokens = bm25s.tokenize([code_friendly_text(query)], show_progress=False)

    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)

    sources: List[MinimalSource] = []
    for i in range(results.shape[1]):
        chunk_data = results[0, i]
        sources.append(
            MinimalSource(
                file_path=chunk_data["file_path"],
                first_character_index=chunk_data["first_character_index"],
                last_character_index=chunk_data["last_character_index"],
            )
        )

    if use_cache:
        query_cache.set(cache_key, sources)
    return sources