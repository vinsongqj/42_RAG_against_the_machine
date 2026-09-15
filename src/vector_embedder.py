from typing import Any
from chromadb.utils import embedding_functions


def get_embedding_function() -> Any:
    return embedding_functions.DefaultEmbeddingFunction()
