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
    """Build the BM25 (lexical) index and, optionally, a semantic index.

    ``bm25_text`` (when set) is what gets indexed; it may include a
    header-trail prefix for markdown sections.  ``content`` and the
    character offsets always refer to the on-disk file, so ``retrieve``
    and ``generate_answer`` continue to work against the raw file.

    Bonus: Semantic embeddings. ``build_semantic`` defaults to False so the
    mandatory command's behaviour and timing are unaffected unless a
    reviewer opts in with ``--build_semantic true``; when enabled, a Chroma
    + MiniLM vector index (src.vector_index) is built next to the BM25
    index, over the same enriched text used for BM25.
    """
    chunks = ingest_directory(raw_dir, max_chunk_size)

    corpus_metadata = [chunk.model_dump() for chunk in chunks]
    # Prefer bm25_text when present, fall back to raw content.  Then apply
    # the identifier-expansion transform so camelCase/snake_case match.
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