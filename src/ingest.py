"""
Reads files from a directory and splits them into chunks for indexing.
"""

from pathlib import Path
from typing import List
from tqdm import tqdm
from src.models import Chunk
from src.chunk import chunk_generic, chunk_markdown, chunk_python


def ingest_directory(target_dir: str, chunk_size: int = 1000) -> List[Chunk]:
    """
    Recursively explores a directory and chunks every file. Files in
    hidden folders, empty files and files that cannot be read are skipped.

    Uses 3 different chunking methods (Refer to chunk.py for more info):
    - chunk_python: For Python files.
    - chunk_markdown: For Markdown files.
    - chunk_generic: For files other than Python and Markdown.

    Args:
        - target_dir: Directory to search recursively.
        - chunk_size: Max character limit of a chunk.

    Returns:
    Chunks from all files.

    Raises:
        - FileNotFoundError: If target_dir does not exist.
        - NotADirectoryError: If target_dir is a file and not a directory.
        - RuntimeError: If target_dir cannot be accessed.
    """
    target_path = Path(target_dir).resolve()  # Get absolute path
    if not target_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {target_path}")
    if not target_path.is_dir():
        raise NotADirectoryError("Expected a directory but got a file:"
                                 f" {target_path}")

    chunks: List[Chunk] = []
    # Recursively search all directories starting from target_path
    try:
        files = list(target_path.rglob("*"))
    except OSError as e:
        raise RuntimeError(f"Could not scan {target_path}: {e}") from e

    skipped = 0

    for file_path in tqdm(files, desc="Ingesting files"):
        try:
            if not file_path.is_file():
                continue
            # Breaks file path into tuple of parts and slices off file name
            # Skips file if any part indicates a hidden folder
            if any(part.startswith(".") for part in file_path.parts[:-1]):
                continue

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content:
                continue

            try:
                relative_path = str(file_path.relative_to(Path.cwd()))
            except ValueError:
                relative_path = str(file_path)

            if file_path.suffix == ".py":
                file_chunks = chunk_python(content, relative_path, chunk_size)
            elif file_path.suffix in (".md", ".markdown"):
                file_chunks = chunk_markdown(content, relative_path,
                                             chunk_size)
            else:
                file_chunks = chunk_generic(content, relative_path, chunk_size)
            chunks.extend(file_chunks)

        except Exception:
            skipped += 1
            tqdm.write(f"Skipping {file_path}")
            continue

    print(f"Total chunks created: {len(chunks)}")

    if skipped:
        print(f"Skipped {skipped} files due to errors.")
    return chunks
