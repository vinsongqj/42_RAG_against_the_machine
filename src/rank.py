import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import bm25s
from src.models import MinimalSource
from src.semantic import semantic_search


# Splits strings like "CamelCase"
_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
# Splits strings like "camelCase"
_CAMEL_LOWER = re.compile(r"([a-z0-9])([A-Z])")
# File extensions considered documentation for boosting
_DOC_EXTENSIONS = (".md", ".markdown", ".rst", ".txt")
# Matches queries like "What is X / define X"
_CONCEPTUAL_QUERY = re.compile(
    r"^\s*(what\s+(is|are)\b|define\b|explain\s+what\b)",
    re.IGNORECASE,
)


def split_identifiers(text: str) -> str:
    """
    Splits code names into separate words.
    (e.g. camelCase -> camel Case and snake_case -> snake case)
    Used when building the index and searching so queries and
    chunks match.

    Args:
        - text: The text to convert.

    Returns:
        The text with code names split into words.
    """
    text = _CAMEL_ACRONYM.sub(r"\1 \2", text)
    text = _CAMEL_LOWER.sub(r"\1 \2", text)
    return text.replace("_", " ")


def is_documentation(file_path: str) -> bool:
    """
    Checks whether a file is considered documentation, and
    whether it has a folder named "docs" in its file path.

    Args:
        - file_path: The path to the file.

    Returns:
        True if file is considered documentation.
    """
    path = Path(file_path)
    return path.suffix.lower() in _DOC_EXTENSIONS and "docs" in path.parts


def is_conceptual_query(query: str) -> bool:
    """
    Checks whether a query expects an explanation, such as
    "What is X" or "Define X".

    Args:
        - query: The search query.

    Returns:
        True if the query looks like it expects and explanation.
    """
    return bool(_CONCEPTUAL_QUERY.match(query))


def bm25_search(retriever: Any, query: str, k: int,
                doc_boost: float = 1.0) -> List[MinimalSource]:
    """
    Searches the BM25 keyword index. If the is_conceptual() is true,
    documentation files get doc_boost multiplier to their score. To let
    documentation files near the boundary of top k move up, extra candidates
    are fetched first when a boost is applied.

    If anything goes wrong, a warning is printed and an empty list is
    returned without raising an error.

    Args:
        - retriever: The loaded BM25 index.
        - query: The search query.
        - k: The number of sources to return.
        - doc_boost: Score multiplier for documentation files. 1.0 to turn off.

    Returns:
        Up to k sources, best match first. Empty if query is blank, k is 0
        or less, the index is empty or the search fails.
    """
    if k <= 0 or not query or not query.strip():
        return []
    try:
        corpus_size = len(retriever.corpus)
        if corpus_size == 0:
            return []
        query_tokens = bm25s.tokenize([split_identifiers(query)],
                                      show_progress=False)
        boost = doc_boost if is_conceptual_query(query) else 1.0
        # max must be >= 30, min returns k if corpus size > k and vice versa
        extra_candidates = min(corpus_size, k if boost == 1.0
                               else max(k * 4, 30))
        results, scores = retriever.retrieve(query_tokens, k=extra_candidates,
                                             show_progress=False)
        boosted = []
        for i in range(results.shape[1]):
            chunk_data = results[0, i]
            score = (float(scores[0, i]) *
                     (boost if is_documentation(chunk_data["file_path"])
                      else 1.0))
            boosted.append((score, chunk_data))
        boosted.sort(key=lambda pair: pair[0], reverse=True)

        return [
            MinimalSource(
                file_path=chunk_data["file_path"],
                first_character_index=chunk_data["first_character_index"],
                last_character_index=chunk_data["last_character_index"]
                ) for _, chunk_data in boosted[:k]]

    except Exception as e:
        print(f"Bm25 search failed for {query!r} ({type(e).__name__}: {e})",
              file=sys.stderr)
        return []


def _source_key(source: MinimalSource) -> Tuple[str, int, int]:
    """
    Identifies a source by its file and character range, so the same chunk
    found by different searches can be recognized.

    Args:
        - source: The source to identify.

    Returns:
        The file path, start and end of the source.
    """
    return (source.file_path, source.first_character_index,
            source.last_character_index)


def reciprocal_rank_fusion(rankings: List[List[MinimalSource]],
                           rrf_constant: int = 60) -> List[MinimalSource]:
    """
    Combines several ranked lists into one.

    The higher a chunk ranks in each list, the more points it gets. A chunk
    found by more than one search has higher ranking. Positions are compared
    while scores are ignored since scores from different search methods can't
    be directly compared.

    Args:
        - rankings: The ranked lists to combine, best match first in each.
        - rrf_constant: Smoothing value, Higher values make the top positions
                        count for less.

    Returns:
        Every source from all list without duplicates, with best match first.
    """
    scores: Dict[Tuple[str, int, int], float] = defaultdict(float)
    first_seen: Dict[Tuple[str, int, int], MinimalSource] = {}

    for ranking in rankings:
        for rank, source in enumerate(ranking):
            key = _source_key(source)
            scores[key] += 1.0 / (rrf_constant + rank + 1)
            # setdefault() prevents duplicating the source data
            # if file is encountered in another list
            first_seen.setdefault(key, source)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [first_seen[key] for key in ranked_keys]


def hybrid_search(retriever: Any, query: str, k: int, index_dir: str,
                  doc_boost: float = 1.0) -> List[MinimalSource]:
    """
    Searches by keyword and meaning, then combines the results.

    Each search fetches extra candidates (3 x k, at least 20) before
    they are merged with reciprocal rank fusion and the top k are returned.
    If semantic search fails, a message is printed and only keyword result
    are used.

    Args:
        - retriever: The loaded BM25 index.
        - query: The search query.
        - k: The number of sources to return.
        - index_dir: Direcotry of the built semantic index.
        - doc_boost: Score multiplier for documentation files. Only
                     affects keyword search.

    Returns:
        Up to k sources, best match first.
    """
    if k <= 0:
        return []

    extra_candidates = max(k * 3, 20)
    lexical = bm25_search(retriever, query, extra_candidates,
                          doc_boost=doc_boost)
    try:
        semantic = semantic_search(query, extra_candidates, index_dir)
    except Exception as e:
        print(f"Semantic search unavailable ({e}), using BM25 only ranking.")
        semantic = []

    fused = reciprocal_rank_fusion([lexical, semantic])
    return fused[:k]
