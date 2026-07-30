import bm25s
from pathlib import Path
from typing import List
from src.models import MinimalSource
from src.cache import query_cache

_retriever_cache = None
_index_dir_cache = None


def _get_retriever(index_dir: str = "data/processed"):
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


def retrieve(query: str, k: int = 5, index_dir: str = "data/processed") -> List[MinimalSource]:
    cache_key = f"{query}_{k}_{index_dir}"
    cached = query_cache.get(cache_key)
    if cached is not None:
        return cached

    retriever = _get_retriever(index_dir)
    # Suppress progress bars
    query_tokens = bm25s.tokenize([query], show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)

    sources = []
    for i in range(results.shape[1]):
        chunk_data = results[0, i]
        sources.append(
            MinimalSource(
                file_path=chunk_data["file_path"],
                first_character_index=chunk_data["first_character_index"],
                last_character_index=chunk_data["last_character_index"]
            )
        )

    query_cache.set(cache_key, sources)
    return sources