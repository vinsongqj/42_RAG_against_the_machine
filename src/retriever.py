from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import bm25s
from src.cache import query_cache
from src.chunker import code_friendly_text
from src.models import MinimalSource
from vector_indexer import semantic_search

RetrievalMethod = str
_VALID_METHODS = ("bm25", "semantic", "hybrid")

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


def _bm25_search(query: str, k: int, index_dir: str) -> List[MinimalSource]:
    if k <= 0:
        return []
    retriever = _get_retriever(index_dir)
    k = min(k, len(retriever.corpus))
    if k <= 0:
        return []

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
    return sources


def _source_key(source: MinimalSource) -> Tuple[str, int, int]:
    return (source.file_path, source.first_character_index, source.last_character_index)


def _reciprocal_rank_fusion(
    rankings: List[List[MinimalSource]],
    rrf_constant: int = 60,
) -> List[MinimalSource]:
    scores: Dict[Tuple[str, int, int], float] = defaultdict(float)
    first_seen: Dict[Tuple[str, int, int], MinimalSource] = {}

    for ranking in rankings:
        for rank, source in enumerate(ranking):
            key = _source_key(source)
            scores[key] += 1.0 / (rrf_constant + rank + 1)
            first_seen.setdefault(key, source)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [first_seen[key] for key in ranked_keys]


def _hybrid_search(query: str, k: int, index_dir: str) -> List[MinimalSource]:
    if k <= 0:
        return []
    pool_size = max(k * 3, 20)
    lexical = _bm25_search(query, pool_size, index_dir)
    semantic = semantic_search(query, pool_size, index_dir)
    fused = _reciprocal_rank_fusion([lexical, semantic])
    return fused[:k]


def retrieve(
    query: str,
    k: int = 5,
    index_dir: str = "data/processed",
    method: RetrievalMethod = "bm25",
    use_cache: bool = True,
) -> List[MinimalSource]:
    
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}; expected one of {_VALID_METHODS}")

    cache_key = f"{query}::{k}::{index_dir}::{method}"
    if use_cache:
        cached = query_cache.get(cache_key)
        if cached is not None:
            return cast(List[MinimalSource], cached)

    if method == "bm25":
        sources = _bm25_search(query, k, index_dir)
    elif method == "semantic":
        sources = semantic_search(query, k, index_dir)
    else:
        sources = _hybrid_search(query, k, index_dir)

    if use_cache:
        query_cache.set(cache_key, sources)
    return sources
