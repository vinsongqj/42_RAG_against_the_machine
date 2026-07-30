import bm25s
from pathlib import Path
from src.ingester import ingest_directory


def build_index(raw_dir: str = "data/raw",
                processed_dir: str = "data/processed",
                max_chunk_size: int = 2000,
                k1: float = 1.5,  # Added BM25 parameters
                b: float = 0.75):
    """
    Build BM25 index with tunable parameters.
    Default: k1=1.5, b=0.75 (good for code/text)
    """
    chunks = ingest_directory(raw_dir, max_chunk_size)
    corpus_texts = [chunk.content for chunk in chunks]
    corpus_metadata = [chunk.model_dump() for chunk in chunks]
    
    # Tokenize
    corpus_tokens = bm25s.tokenize(corpus_texts)
    
    # Build with tuned parameters
    retriever = bm25s.BM25(
        corpus=corpus_metadata,
    )
    retriever.index(corpus_tokens)
    
    # Save
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(processed_path / "bm25_index"), corpus=corpus_metadata)
    
    print(f"Index saved to {processed_path / 'bm25_index'}")
    print(f"Indexed {len(chunks)} chunks")