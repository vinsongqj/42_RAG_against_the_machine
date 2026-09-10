from pathlib import Path
from typing import List
from tqdm import tqdm
from src.models import CodeChunk
from src.chunker import chunk_markdown, chunk_python


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
    """Generic chunking for prose-ish text (rst, txt, yaml, Dockerfile, ...).

    Sentence- and paragraph-level separators before falling back to word
    and character splits.  The previous separator list put a 4-space run
    (``"    "``) ahead of ``" "`` which is a code-indent heuristic and only
    hurt prose files; it has been removed.
    """
    from src.chunker import RecursiveCharacterTextSplitter

    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(250, chunk_size // 4),
        separators=separators,
    )

    texts = splitter.split_text(content)
    chunks: List[CodeChunk] = []
    position = 0

    for text in texts:
        start = content.find(text, position)
        if start == -1:
            start = position
        end = start + len(text)
        if end - start > chunk_size:
            end = start + chunk_size
            text = content[start:end]

        chunks.append(
            CodeChunk(
                file_path=file_path,
                content=text,
                first_character_index=start,
                last_character_index=end,
            )
        )
        position = end

    return chunks