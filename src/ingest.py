from pathlib import Path
from typing import List
from tqdm import tqdm
from src.models import CodeChunk
from src.chunk import chunk_markdown, chunk_python, RecursiveCharacterTextSplitter, _to_chunks


_SKIP_DIR_NAMES = {
    "__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env",
    "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "build", "dist", ".tox", ".idea", ".vscode",
}


def ingest_directory(target_dir: str, chunk_size: int = 1000) -> List[CodeChunk]:
    target_path = Path(target_dir).resolve()
    chunks: List[CodeChunk] = []
    files = list(target_path.rglob("*"))

    for file_path in tqdm(files, desc="Ingesting files"):
        if not file_path.is_file():
            continue

        if any(
            part.startswith(".") or part in _SKIP_DIR_NAMES
            for part in file_path.parts[:-1]
        ):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        if not content:
            continue

        rel_path = str(file_path.relative_to(Path.cwd()))

        if file_path.suffix == ".py":
            file_chunks = chunk_python(content, rel_path, chunk_size)
        elif file_path.suffix in (".md", ".markdown"):
            file_chunks = chunk_markdown(content, rel_path, chunk_size)
        else:
            file_chunks = chunk_generic(content, rel_path, chunk_size)

        chunks.extend(file_chunks)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def chunk_generic(content: str, file_path: str, chunk_size: int = 1000) -> List[CodeChunk]:
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(250, chunk_size // 4),
        separators=separators,
    )

    texts = splitter.split_text(content)
    return _to_chunks(content, file_path, splitter, texts=texts)
