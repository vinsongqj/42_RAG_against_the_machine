"""CPU-friendly MiniLM embeddings (Bonus: Semantic embeddings).

Wraps sentence-transformers' ``all-MiniLM-L6-v2`` -- small (~80MB) and fast
on CPU, the exact lightweight model named in the project brief. The model
is loaded once per process (module-level cache, mirroring the BM25
retriever's cache in src.retriever) and reused for every call.

This module only turns text into vectors; it has no knowledge of Chroma or
BM25. src.vector_index calls embed_texts() when building the vector index
and embed_query() when searching it.
"""

from typing import Any, List, Optional, cast

from sentence_transformers import SentenceTransformer

MODEL_ID = "all-MiniLM-L6-v2"

_model: Optional[Any] = None


def _get_model() -> Any:
    global _model
    if _model is None:
        print(f"Loading embedding model {MODEL_ID}...")
        _model = SentenceTransformer(MODEL_ID, device="cpu")
        print("Embedding model loaded!")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts (e.g. chunk contents) into dense vectors."""
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return cast(List[List[float]], vectors.tolist())


def embed_query(query: str) -> List[float]:
    """Embed a single query string into a dense vector."""
    return embed_texts([query])[0]