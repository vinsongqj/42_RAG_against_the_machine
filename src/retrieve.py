from pathlib import Path
from typing import Any, List
import bm25s
from src.cache import query_cache
from src.fingerprint import dir_fingerprint, FingerprintedCache
from src.models import MinimalSource
from src.rank import bm25_search, hybrid_search
from src.semantic_embedding import semantic_search

RetrievalMethod = str
_VALID_METHODS = ("bm25", "semantic", "hybrid")
DEFAULT_DOC_BOOST = 1.3


def _load_bm25(index_dir: str) -> Any:
    index_path = Path(index_dir) / "bm25_index"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    print(f"Loading index from {index_path}...")
    retriever = bm25s.BM25.load(str(index_path), load_corpus=True)
    print("Index loaded successfully!")
    return retriever


_retriever_cache = FingerprintedCache(_load_bm25)


def _get_retriever(index_dir: str = "data/processed") -> Any:
    return _retriever_cache.get(index_dir, Path(index_dir) / "bm25_index")


def preload_retriever(index_dir: str = "data/processed") -> None:
    _get_retriever(index_dir)


def _index_version(index_dir: str, method: RetrievalMethod) -> str:
    paths = []
    if method in ("bm25", "hybrid"):
        paths.append(Path(index_dir) / "bm25_index")
    if method in ("semantic", "hybrid"):
        paths.append(Path(index_dir) / "chroma")
    stamps = [dir_fingerprint(p) for p in paths]
    return "-".join(stamps) if stamps else "0"


def retrieve(
    query: str,
    k: int = 5,
    index_dir: str = "data/processed",
    method: RetrievalMethod = "bm25",
    use_cache: bool = True,
    doc_boost: float = DEFAULT_DOC_BOOST,
) -> List[MinimalSource]:
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}; expected one of {_VALID_METHODS}")

    index_version = _index_version(index_dir, method)
    cache_key = f"{query}::{k}::{index_dir}::{method}::{doc_boost}::{index_version}"
    if use_cache:
        cached = query_cache.get(cache_key)
        if cached is not None:
            return [MinimalSource(**d) for d in cached]

    if method == "bm25":
        retriever = _get_retriever(index_dir)
        sources = bm25_search(retriever, query, k, doc_boost=doc_boost)
    elif method == "semantic":
        sources = semantic_search(query, k, index_dir)
    else:
        retriever = _get_retriever(index_dir)
        sources = hybrid_search(retriever, query, k, index_dir, doc_boost=doc_boost)

    if use_cache:
        query_cache.set(cache_key, [s.model_dump() for s in sources])
    return sources
