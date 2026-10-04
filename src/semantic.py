"""
Stores chunks as embeddings and finds the ones closest in meaning to
the query.
"""

from pathlib import Path
from typing import Any, Dict, List
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from src.version import VersionCache
from src.models import Chunk, MinimalSource


_COLLECTION_NAME = "rag_chunks"
# How many chunks to embed per batch
_ADD_BATCH_SIZE = 256
# Cache of Chroma clients keyed by index directory
_client_cache: Dict[str, Any] = {}


def get_embedding_function() -> Any:
    """
    Returns the model that turns text into embeddings (lists
    of numbers that capture meaning). The same model must be used when
    building the index and when searching otherwise results won't match.

    Returns:
        The embedding function.
    """
    return embedding_functions.DefaultEmbeddingFunction()


def _get_client(index_dir: str) -> Any:
    """
    Opens the Chroma database inside the index folder, reusing it if it is
    already open.

    Args:
        - index_dir: Directory of the built index. The database is stored in a
                     /chroma folder inside it and created if missing.

    Returns:
        The Chroma client.

    Raises:
        RuntimeError: If the database fails to open.
    """
    if index_dir not in _client_cache:
        chroma_path = Path(index_dir) / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        try:
            _client_cache[index_dir] = chromadb.PersistentClient(
                path=str(chroma_path))
        except Exception as e:
            raise RuntimeError(f"Could not open Chroma DB at {chroma_path} "
                               f"({type(e).__name__}: {e})") from e
        return _client_cache[index_dir]


def _load_collection(index_dir: str) -> Any:
    """
    Loads the saved semantic index from the disk.
    Called by the cache whenever the index needs to be loaded
    or reloaded.

    Args:
        - index_dir: Directory of the built index.

    Returns:
        The Chroma collection holding the embedded chunks.

    Raises:
        FileNotFoundError: If the semantic index hasn't been built.
    """
    client = _get_client(index_dir)
    try:
        return client.get_collection(
            _COLLECTION_NAME,
            embedding_function=get_embedding_function()
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Semantic index not found under {index_dir}; "
            "run `index` with semantic embeddings enabled first."
        ) from e


_collection_cache = VersionCache(_load_collection)


def _delete_collection(client: Any) -> None:
    """
    Deletes the semantic collection if it exists. Does nothing if it doesn't.

    Args:
        - client: The Chroma client to delete from.
    """
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass


def build_vector_index(chunks: List[Chunk], texts: List[str],
                       processed_dir: str) -> None:
    """
    Builds the semantic index and replaces any existing one.
    Each chunk's text is converted into an embedding and saved
    with its file path and character range so a search result
    can be turned back into a source. If build fails, the index
    is deleted so it can't be searched by mistake.

    Args:
        - chunks: The chunks to index.
        - texts: The text to embed for each chunk in the same order.
        - processed_dir: Directory where the index is saved.

    Raises:
        - ValueError: If chunks and texts differ in length.
        - RuntimeError: If the Chroma database can't be opened or
                        build fails.
    """
    if len(chunks) != len(texts):
        raise ValueError(f"Mismatch found: got {len(chunks)} chunks "
                         f"but {len(texts)} texts")
    client = _get_client(processed_dir)
    _collection_cache.invalidate()
    _delete_collection(client)
    try:
        collection = client.create_collection(
            _COLLECTION_NAME,
            embedding_function=get_embedding_function()
        )
        for start in tqdm(range(0, len(chunks), _ADD_BATCH_SIZE),
                          desc="Embedding chunks"):
            batch = chunks[start:start + _ADD_BATCH_SIZE]
            collection.add(
                ids=[str(start + i) for i in range(len(batch))],
                documents=texts[start:start + _ADD_BATCH_SIZE],
                metadatas=[
                    {
                        "file_path": c.file_path,
                        "first_character_index": c.first_character_index,
                        "last_character_index": c.last_character_index
                    }
                    for c in batch
                ]
            )
    except Exception as e:
        _delete_collection(client)
        raise RuntimeError(
            f"Building the semantic index failed ({type(e).__name__}: {e}). "
            "The partial collection was removed."
        ) from e

    _collection_cache.set(processed_dir, collection,
                          Path(processed_dir) / "chroma")
    print(f"Semantic index saved to {Path(processed_dir) / 'chroma'}")
    print(f"Embedded {len(chunks)} chunks")


def _get_collection(index_dir: str) -> Any:
    """
    Returns the semantic index, loading it only if index files have changed.

    Args:
        - index_dir: Directory of the built index.

    Returns:
        The Chroma collection holding the embedded chunks.

    Raises:
        FileNotFoundError: If the semantic index hasn't been built.
    """
    return _collection_cache.get(index_dir, Path(index_dir) / "chroma")


def preload_collection(index_dir: str = "data/processed") -> None:
    """
    Loads the semantic index ahead of the first search.
    Used by the API health check to check if the index exists.

    Args:
        - index_dir: Directory of the built index.

    Raises:
        FileNotFoundError: If the semantic index hasn't been built.
    """
    _get_collection(index_dir)


def semantic_search(query: str, k: int = 5,
                    index_dir: str = "data/processed") -> List[MinimalSource]:
    """
    Finds the chunks closest in meaning to the query.
    The query is turned into an embedding and compared to the stored ones.

    Args:
        - query: The search query.
        - k: The number of sources to return.
        - index_dir: Directory of the built index.

    Returns:
        Up to k sources, closest match first. Empty if k is 0 or less, the
        query is blank, or the index has no chunks.

    Raises:
        - FileNotFoundError: If the semantic index hasn't been built.
        - RuntimeError: If the semantic search fails.
    """
    if k <= 0:
        return []
    collection = _get_collection(index_dir)
    if not query or not query.strip():
        return []
    try:
        n_results = min(k, collection.count())
        if n_results <= 0:
            return []
        results = collection.query(query_texts=[query], n_results=n_results)
    except Exception as e:
        raise RuntimeError(f"Semantic search failed ({type(e).__name__}: {e})"
                           ) from e

    sources = []
    metadatas: List[Dict[str, Any]] = (results.get("metadatas") or [[]])[0]
    for meta in metadatas:
        if not meta:
            continue
        sources.append(
            MinimalSource(
                file_path=str(meta["file_path"]),
                first_character_index=int(meta["first_character_index"]),
                last_character_index=int(meta["last_character_index"])
            )
        )
    return sources
