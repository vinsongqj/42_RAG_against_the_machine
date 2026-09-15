import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import bm25s
from src.cache import query_cache
from src.chunker import code_friendly_text
from src.models import MinimalSource
from src.vector_indexer import semantic_search

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


_DOC_EXTENSIONS = (".md", ".markdown", ".rst", ".txt")


def _is_doc_path(file_path: str) -> bool:
    path = Path(file_path)
    return path.suffix.lower() in _DOC_EXTENSIONS and "docs" in path.parts


_CONCEPTUAL_QUERY_RE = re.compile(
    r"^\s*(what\s+(is|are)\b|define\b|explain\s+what\b)",
    re.IGNORECASE,
)


def _is_conceptual_query(query: str) -> bool:
    return bool(_CONCEPTUAL_QUERY_RE.match(query))


def _sources_from_bm25_results(results: Any, row: int = 0) -> List[MinimalSource]:
    sources: List[MinimalSource] = []
    for i in range(results.shape[1]):
        chunk_data = results[row, i]
        sources.append(
            MinimalSource(
                file_path=chunk_data["file_path"],
                first_character_index=chunk_data["first_character_index"],
                last_character_index=chunk_data["last_character_index"],
            )
        )
    return sources


def _bm25_search(query: str, k: int, index_dir: str, doc_boost: float = 1.0) -> List[MinimalSource]:
    if k <= 0:
        return []
    retriever = _get_retriever(index_dir)
    corpus_size = len(retriever.corpus)

    query_tokens = bm25s.tokenize([code_friendly_text(query)], show_progress=False)

    effective_doc_boost = doc_boost if _is_conceptual_query(query) else 1.0

    if effective_doc_boost == 1.0:
        k = min(k, corpus_size)
        if k <= 0:
            return []
        results, _ = retriever.retrieve(query_tokens, k=k, show_progress=False)
        return _sources_from_bm25_results(results)

    pool_k = min(max(k * 4, 30), corpus_size)
    if pool_k <= 0:
        return []
    results, scores = retriever.retrieve(query_tokens, k=pool_k, show_progress=False)

    boosted = []
    for i in range(results.shape[1]):
        chunk_data = results[0, i]
        score = float(scores[0, i]) * (effective_doc_boost if _is_doc_path(chunk_data["file_path"]) else 1.0)
        boosted.append((score, chunk_data))
    boosted.sort(key=lambda pair: pair[0], reverse=True)

    sources = [
        MinimalSource(
            file_path=chunk_data["file_path"],
            first_character_index=chunk_data["first_character_index"],
            last_character_index=chunk_data["last_character_index"],
        )
        for _, chunk_data in boosted[:k]
    ]
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


def _hybrid_search(query: str, k: int, index_dir: str, doc_boost: float = 1.0) -> List[MinimalSource]:
    if k <= 0:
        return []

    pool_size = max(k * 3, 20)
    lexical = _bm25_search(query, pool_size, index_dir, doc_boost=doc_boost)
    try:
        semantic = semantic_search(query, pool_size, index_dir)
    except Exception as e:
        print(f"Semantic search unavailable ({e}); using BM25-only ranking for this query.")
        semantic = []

    fused = _reciprocal_rank_fusion([lexical, semantic])
    return fused[:k]


def retrieve(
    query: str,
    k: int = 5,
    index_dir: str = "data/processed",
    method: RetrievalMethod = "bm25",
    use_cache: bool = True,
    doc_boost: float = 1.3,
) -> List[MinimalSource]:

    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}; expected one of {_VALID_METHODS}")

    cache_key = f"{query}::{k}::{index_dir}::{method}::{doc_boost}"
    if use_cache:
        cached = query_cache.get(cache_key)
        if cached is not None:
            return cast(List[MinimalSource], cached)

    if method == "bm25":
        sources = _bm25_search(query, k, index_dir, doc_boost=doc_boost)
    elif method == "semantic":
        sources = semantic_search(query, k, index_dir)
    else:
        sources = _hybrid_search(query, k, index_dir, doc_boost=doc_boost)

    if use_cache:
        query_cache.set(cache_key, sources)
    return sources
