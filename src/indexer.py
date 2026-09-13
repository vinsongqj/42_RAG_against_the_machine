import bm25s
from pathlib import Path
from src.ingester import ingest_directory
from src.chunker import code_friendly_text
from vector_indexer import build_vector_index


def build_index(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    max_chunk_size: int = 1000,
    k1: float = 1.5,
    b: float = 0.75,
    build_semantic: bool = False,
) -> None:
   
    chunks = ingest_directory(raw_dir, max_chunk_size)

    corpus_metadata = [chunk.model_dump() for chunk in chunks]

    corpus_texts = [
        code_friendly_text(c.bm25_text if c.bm25_text else c.content)
        for c in chunks
    ]

    corpus_tokens = bm25s.tokenize(corpus_texts)

    retriever = bm25s.BM25(corpus=corpus_metadata, k1=k1, b=b)
    retriever.index(corpus_tokens)

    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(processed_path / "bm25_index"), corpus=corpus_metadata)

    print(f"Index saved to {processed_path / 'bm25_index'}")
    print(f"Indexed {len(chunks)} chunks")

    if build_semantic:
        build_vector_index(chunks, corpus_texts, processed_dir)
