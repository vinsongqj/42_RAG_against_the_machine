from pathlib import Path
from typing import Any, Dict, List
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from src.fingerprint import FingerprintedCache
from src.models import CodeChunk, MinimalSource

_COLLECTION_NAME = "rag_chunks"
_ADD_BATCH_SIZE = 256

_client_cache: Dict[str, Any] = {}


def get_embedding_function() -> Any:
    return embedding_functions.DefaultEmbeddingFunction()


def _get_client(index_dir: str) -> Any:
    if index_dir not in _client_cache:
        chroma_path = Path(index_dir) / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        _client_cache[index_dir] = chromadb.PersistentClient(path=str(chroma_path))
    return _client_cache[index_dir]


def _load_collection(index_dir: str) -> Any:
    client = _get_client(index_dir)
    try:
        return client.get_collection(
            _COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Semantic index not found under {index_dir}; "
            "run `index` with semantic embeddings enabled first."
        ) from e


_collection_cache = FingerprintedCache(_load_collection)


def build_vector_index(
    chunks: List[CodeChunk],
    texts: List[str],
    processed_dir: str,
) -> None:
    client = _get_client(processed_dir)
    _collection_cache.invalidate()
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        _COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )

    for start in tqdm(range(0, len(chunks), _ADD_BATCH_SIZE), desc="Embedding chunks"):
        batch_chunks = chunks[start:start + _ADD_BATCH_SIZE]
        batch_texts = texts[start:start + _ADD_BATCH_SIZE]
        ids = [str(start + i) for i in range(len(batch_chunks))]
        metadatas = [
            {
                "file_path": c.file_path,
                "first_character_index": c.first_character_index,
                "last_character_index": c.last_character_index,
            }
            for c in batch_chunks
        ]
        collection.add(ids=ids, documents=batch_texts, metadatas=metadatas)

    _collection_cache.set(processed_dir, collection, Path(processed_dir) / "chroma")
    print(f"Semantic index saved to {Path(processed_dir) / 'chroma'}")
    print(f"Embedded {len(chunks)} chunks")


def _get_collection(index_dir: str) -> Any:
    return _collection_cache.get(index_dir, Path(index_dir) / "chroma")


def preload_collection(index_dir: str = "data/processed") -> None:
    _get_collection(index_dir)


def semantic_search(query: str, k: int = 5, index_dir: str = "data/processed") -> List[MinimalSource]:
    if k <= 0:
        return []
    collection = _get_collection(index_dir)
    results = collection.query(query_texts=[query], n_results=k)

    sources = []
    metadatas_by_query = results.get("metadatas") or [[]]
    metadatas: List[Dict[str, Any]] = metadatas_by_query[0] if metadatas_by_query else []
    for meta in metadatas:
        sources.append(
            MinimalSource(
                file_path=str(meta["file_path"]),
                first_character_index=int(meta["first_character_index"]),
                last_character_index=int(meta["last_character_index"]),
            )
        )
    return sources
