"""CPU-friendly MiniLM embeddings (Bonus: Semantic embeddings).

Uses ChromaDB's built-in DefaultEmbeddingFunction, which runs
all-MiniLM-L6-v2 through onnxruntime -- the exact model named in the
project brief, but without a PyTorch/sentence-transformers dependency.

The earlier version of this module wrapped sentence-transformers directly,
which works fine but transitively pulls in both `torch` and `transformers`
as real installed dependencies (confirmed via `pip show sentence-transformers`)
even though generator.py no longer needs either for answer generation.
Routing through Chroma's own ONNX-backed default avoids that entirely.
"""

from typing import Any

from chromadb.utils import embedding_functions


def get_embedding_function() -> Any:
    """all-MiniLM-L6-v2 via onnxruntime -- Chroma's CPU-friendly default.

    Returned object is a Chroma EmbeddingFunction: a callable that Chroma
    invokes itself when you pass `documents=`/`query_texts=` to a
    collection, so callers here never touch raw vectors directly.
    """
    return embedding_functions.DefaultEmbeddingFunction()