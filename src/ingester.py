from pathlib import Path
from typing import List
from tqdm import tqdm
from src.models import CodeChunk
from src.chunker import chunk_markdown, chunk_python


def ingest_directory(target_dir: str, chunk_size: int = 2000) -> List[CodeChunk]:
    """
    Read all files from target_dir, split into chunks, and return CodeChunk list.
    File paths are stored relative to the current working directory.
    """
    target_path = Path(target_dir).resolve()
    chunks = []
    files = list(target_path.rglob("*"))

    for file_path in tqdm(files, desc="Ingesting files"):
        if not file_path.is_file():
            continue
        # Skip hidden or special dirs
        if any(part.startswith((".", "__")) for part in file_path.parts):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

        # Skip empty files
        if not content:
            continue

        rel_path = str(file_path.relative_to(Path.cwd()))

        # Chunk based on file type
        if file_path.suffix == ".py":
            file_chunks = chunk_python(content, rel_path, chunk_size)
        elif file_path.suffix in (".md", ".markdown"):
            file_chunks = chunk_markdown(content, rel_path, chunk_size)
        else:
            # For all other files (Dockerfile, jinja, yaml, txt, etc.)
            # Use generic chunking - split by newlines with size limit
            file_chunks = chunk_generic(content, rel_path, chunk_size)
        
        chunks.extend(file_chunks)

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def chunk_generic(content: str, file_path: str, chunk_size: int = 2000) -> List[CodeChunk]:
    """Generic chunking for any text file - splits by newlines."""
    from src.chunker import RecursiveCharacterTextSplitter
    
    # Generic separators for any text file
    separators = ["\n\n", "\n", ". ", " ", "    "]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=separators)
    
    texts = splitter.split_text(content)
    chunks = []
    position = 0

    for text in texts:
        start = content.find(text, position)
        if start == -1:
            start = position
        end = start + len(text)
        
        # Safety check
        if end - start > chunk_size:
            end = start + chunk_size
            text = content[start:end]
        
        chunks.append(
            CodeChunk(
                file_path=file_path,
                content=text,
                first_character_index=start,
                last_character_index=end
            )
        )
        position = end

    return chunks