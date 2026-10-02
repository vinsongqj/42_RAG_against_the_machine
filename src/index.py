"""
Builds the search indexes from a directory of raw source files.
"""

import bm25s
from pathlib import Path
from src.ingest import ingest_directory
from src.rank import split_identifiers
from src.semantic_embedding import build_vector_index
from src.cache import query_cache


def build_index(raw_dir: str = "data/raw",
                processed_dir: str = "data/processed",
                max_chunk_size: int = 1000, k1: float = 1.5, b: float = 0.75,
                build_semantic: bool = False) -> None:
    """
    Chunks the files in raw_dir and builds a BM25 index from them.

    Clears query cache first, then each chunk's text has its identifiers split
    into words before indexing. The index is saved to a bm25_index folder
    inside processed_dir. If build_semantic is True, a vector index is
    also built there.

    Args:
        - raw_dir: Directory of source files to index.
        - processed_dir: Directory where index is saved.
        - max_chunk_size: Max character limit of a chunk.
        - k1: BM25 term frequency saturation. Higher values let repeated words
              add to the score.
        - b: BM25 length normalization, from 0 to 1. Higher values penalize
             long chunks more.
        - build_semantic: Whether or not to build the semantic embedding index.
    """
    query_cache.clear()

    chunks = ingest_directory(raw_dir, max_chunk_size)
    # Pydantic method model_dump() converts Chunk object into dict
    corpus_metadata = [chunk.model_dump() for chunk in chunks]
    corpus_texts = [split_identifiers(chunk.bm25_text if chunk.bm25_text else
                                      chunk.content for chunk in chunks)]
    corpus_tokens = bm25s.tokenize(corpus_texts)
    retriever = bm25s.BM25(corpus=corpus_metadata, k1=k1, b=b)
    retriever.index(corpus_tokens)  # Creates matrix of scores
    processed_path = Path(processed_dir)
    # parents=True creates missing parent folders
    # exist_ok=True doesn't raise errors if folder exists
    processed_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(processed_path / "bm25_index"), corpus=corpus_metadata)

    print(f"Index saved to {processed_path / 'bm25_index'}")
    print(f"Indexed {len(chunks)} chunks")

    if build_semantic:
        build_vector_index(chunks, corpus_texts, processed_dir)
