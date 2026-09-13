"""Semantic (vector) index, built with ChromaDB, that sits next to the
lexical BM25 index from src/indexer.py and src/retriever.py.

Bonus: Semantic embeddings. This module only builds and queries the vector
index on its own -- merging its ranking with BM25's into a single result
list is the separate hybrid-retrieval bonus (see src/retriever.py).

Chunks are embedded with ChromaDB's built-in all-MiniLM-L6-v2 (via
onnxruntime, see src.embedder) and stored in a Chroma PersistentClient
collection under <processed_dir>/chroma. Chroma persists to disk itself,
so a fresh process reuses the on-disk collection instead of recomputing
anything at cold start -- the same caching pattern used for the BM25
retriever. Embedding itself happens inside Chroma (we pass `documents=`
and it calls the attached embedding function), not in this module.
"""

from pathlib import Path
from typing import Any, Dict, List

import chromadb
from tqdm import tqdm

from src.vector_embedder import get_embedding_function
from src.models import CodeChunk, MinimalSource

_COLLECTION_NAME = "rag_chunks"
_ADD_BATCH_SIZE = 256

# One Chroma client per index_dir, kept warm for the life of the process
# (mirrors the BM25 retriever's module-level cache in src.retriever).
_client_cache: Dict[str, Any] = {}


def _get_client(index_dir: str) -> Any:
    if index_dir not in _client_cache:
        chroma_path = Path(index_dir) / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)
        _client_cache[index_dir] = chromadb.PersistentClient(path=str(chroma_path))
    return _client_cache[index_dir]


def build_vector_index(
    chunks: List[CodeChunk],
    texts: List[str],
    processed_dir: str,
) -> None:
    """Embed every chunk and (re)persist the Chroma collection.

    Args:
        chunks: The chunks to embed, as produced by ``src.ingester``.
        texts: The text to embed for each chunk, aligned 1:1 with
            ``chunks``. Callers pass the same enriched text used for BM25
            (header-trail prefix + identifier splitting) so semantic search
            benefits from the same context, not just raw chunk content.
        processed_dir: Directory the persistent Chroma store lives under
            (mirrors the BM25 index's ``processed_dir``).
    """
    client = _get_client(processed_dir)
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet, nothing to drop
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
        # Chroma embeds `documents` itself via the collection's attached
        # embedding function -- no manual embed_texts()/vectors here.
        collection.add(
            ids=ids,
            documents=batch_texts,
            metadatas=metadatas,
        )

    print(f"Semantic index saved to {Path(processed_dir) / 'chroma'}")
    print(f"Embedded {len(chunks)} chunks")


def _get_collection(index_dir: str) -> Any:
    client = _get_client(index_dir)
    try:
        # The embedding function must be reattached on every load -- Chroma
        # needs it to embed the *query* text at search time, not just at
        # build time.
        return client.get_collection(
            _COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Semantic index not found under {index_dir}; "
            "run `index` with semantic embeddings enabled first."
        ) from e


def semantic_search(
    query: str,
    k: int = 5,
    index_dir: str = "data/processed",
) -> List[MinimalSource]:
    """Return the top-k semantically closest chunks for a query."""
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
