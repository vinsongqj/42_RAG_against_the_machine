import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import bm25s
from src.chunk import code_friendly_text
from src.models import MinimalSource
from src.semantic_embedding import semantic_search


_DOC_EXTENSIONS = (".md", ".markdown", ".rst", ".txt")


def is_doc_path(file_path: str) -> bool:
    path = Path(file_path)
    return path.suffix.lower() in _DOC_EXTENSIONS and "docs" in path.parts


_CONCEPTUAL_QUERY_RE = re.compile(
    r"^\s*(what\s+(is|are)\b|define\b|explain\s+what\b)",
    re.IGNORECASE,
)


def is_conceptual_query(query: str) -> bool:
    return bool(_CONCEPTUAL_QUERY_RE.match(query))


def bm25_search(retriever: Any, query: str, k: int,
                doc_boost: float = 1.0) -> List[MinimalSource]:
    if k <= 0:
        return []
    corpus_size = len(retriever.corpus)
    if corpus_size <= 0:
        return []

    query_tokens = bm25s.tokenize([code_friendly_text(query)],
                                  show_progress=False)

    effective_boost = doc_boost if is_conceptual_query(query) else 1.0

    pool_k = min(corpus_size, k if effective_boost == 1.0 else max(k * 4, 30))
    if pool_k <= 0:
        return []
    results, scores = retriever.retrieve(query_tokens, k=pool_k, show_progress=False)

    boosted = []
    for i in range(results.shape[1]):
        chunk_data = results[0, i]
        score = float(scores[0, i]) * (effective_boost if is_doc_path(chunk_data["file_path"]) else 1.0)
        boosted.append((score, chunk_data))
    boosted.sort(key=lambda pair: pair[0], reverse=True)

    return [
        MinimalSource(
            file_path=chunk_data["file_path"],
            first_character_index=chunk_data["first_character_index"],
            last_character_index=chunk_data["last_character_index"],
        )
        for _, chunk_data in boosted[:k]
    ]


def _source_key(source: MinimalSource) -> Tuple[str, int, int]:
    return (source.file_path, source.first_character_index,
            source.last_character_index)


def reciprocal_rank_fusion(
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


def hybrid_search(retriever: Any, query: str, k: int, index_dir: str,
                  doc_boost: float = 1.0) -> List[MinimalSource]:
    if k <= 0:
        return []

    pool_size = max(k * 3, 20)
    lexical = bm25_search(retriever, query, pool_size, doc_boost=doc_boost)
    try:
        semantic = semantic_search(query, pool_size, index_dir)
    except Exception as e:
        print(f"Semantic search unavailable ({e}), using BM25-only ranking for this query.")
        semantic = []
    fused = reciprocal_rank_fusion([lexical, semantic])
    return fused[:k]
