"""
Retrieves the most relevant sources for a query from the built indexes.
"""

from pathlib import Path
from typing import Any, List
import bm25s
from src.models import MinimalSource
from src.rank import bm25_search, hybrid_search
from src.cache import query_cache
from src.version import cached_dir_version, VersionCache
from src.semantic import semantic_search

_VALID_METHODS = ("bm25", "semantic", "hybrid")
DEFAULT_DOC_BOOST = 1.3


def _load_bm25(index_dir: str) -> Any:
    """
    Loads the BM25 index and chunk metadata from disk.
    Reads the full index at every call.
    Only called by the retriever cache if nothing has been
    loaded or the index has been changed.

    Args:
        - index_dir: Directory of the built index.

    Returns:
        The loaded BM25 retriever with chunk metadata.

    Raises:
       - FileNotFoundError: If no BM25 index exists in index_dir.
       - RuntimeError: If the index files exist but cannot be loaded
                       due to corruption/incompleteness.
    """
    index_path = Path(index_dir) / "bm25_index"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    print(f"Loading index from {index_path}...")
    try:
        # load_corpus=True loads saved chunk metadata
        retriever = bm25s.BM25.load(str(index_path), load_corpus=True)
    except Exception as e:
        raise RuntimeError(f"Could not load BM25 index at {index_path}."
                           " Try rebuilding with the index command"
                           ) from e
    print("Index loaded successfully!")
    return retriever


# Holds loaded index
_retriever_cache = VersionCache(_load_bm25)


def _get_retriever(index_dir: str) -> Any:
    """
    Returns the BM25 retriever from memory, only reading from disk when needed.
    Reuses retriever from the previous call unless index_dir differs or index
    files have changed since loading.

    Args:
        - index_dir: Directory of the built index.

    Returns:
        The loaded BM25 retriever.

    Raises:
        - FileNotFoundError: If no BM25 index exists in index_dir.
        - RuntimeError: If the index cannot be loaded and no earlier copy
                        is available to fall back on.
    """
    return _retriever_cache.get(index_dir, Path(index_dir) / "bm25_index")


def preload_retriever(index_dir: str = "data/processed") -> None:
    """
    Loads the BM25 retriever ahead of the first query.

    Args:
        - index_dir: Directory of the built index.

    Raises:
        - FileNotFoundError: If no BM25 index exists in index_dir.
        - RuntimeError: If the index files exist but cannot be loaded.
    """
    _get_retriever(index_dir)


def _index_version(index_dir: str, method: str) -> str:
    """
    Returns a string that changes whenever the index is rebuilt.
    The string is used in the query cache key, so results cached
    before reindexing are never served after.

    Args:
        - index_dir: Directory of the built index.
        - method: Retrieval method which decides which indexes are checked.

    Returns:
        The version of each index used by the method, joined with dashes.
    """
    paths = []
    if method in ("bm25", "hybrid"):
        paths.append(Path(index_dir) / "bm25_index")
    if method in ("semantic", "hybrid"):
        paths.append(Path(index_dir) / "chroma")
    return "-".join(cached_dir_version(p) for p in paths)


def retrieve(query: str, k: int = 5, index_dir: str = "data/processed",
             method: str = "bm25", use_cache: bool = True,
             doc_boost: float = DEFAULT_DOC_BOOST) -> List[MinimalSource]:
    """
    Returns the k most relevant sources for a query.

    Args:
        - query: The search query text.
        - k: The number of sources to return.
        - index_dir: Directory of the built index.
        - method: Retrieval method ("bm25" / "semantic" / "hybrid")
        - use_cache: Whether to use the query cache.
        - doc_boost: Score multiplier for documentation files on
                     Conceptual queries. Unused by semantic method.

    Returns:
        The sources in order of the best match.

    Raises:
        - ValueError: If method is not recognized.
        - FileNotFoundError: If index has not been built.
        - RuntimeError: If an index cannot be loaded, or
                        semantic search fails.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown retrieval method {method!r} - "
                         f"expected one of {_VALID_METHODS}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    index_version = _index_version(index_dir, method)
    cache_key = (
        f"{query}::{k}::{index_dir}::{method}::{doc_boost}::{index_version}"
    )
    # Converts dicts into MinimalSource objects
    if use_cache:
        cached = query_cache.get(cache_key)
        if cached is not None:
            try:
                return [MinimalSource(**d) for d in cached]
            except (TypeError, ValueError):
                pass

    if method == "bm25":
        retriever = _get_retriever(index_dir)
        sources = bm25_search(retriever, query, k, doc_boost=doc_boost)
    elif method == "semantic":
        sources = semantic_search(query, k, index_dir)
    else:
        retriever = _get_retriever(index_dir)
        sources = hybrid_search(retriever, query, k, index_dir,
                                doc_boost=doc_boost)

    # Converts MinimalSource objects into dicts
    if use_cache:
        query_cache.set(cache_key, [s.model_dump() for s in sources])
    return sources
