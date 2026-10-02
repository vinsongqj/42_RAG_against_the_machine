"""
Chunking utilities for the RAG pipeline, used in ingest.py.

This module has 3 chunking utilities:
- chunk_python: For Python files.
- chunk_markdown: For Markdown files.
- chunk_generic: For files other than Python and Markdown.

All return Chunk objects with exact source character spans.
"""

import re
from typing import List, Optional, Tuple
from src.models import Chunk


class RecursiveCharacterTextSplitter:
    """
    Splits text into overlapping chunks. Tries separators from coarsest to
    finest. Splits that fit within chunk size are merged into chunks
    while pieces that are still too long are split again by the next separator.
    """
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None) -> None:
        """
        Initializes the splitter.

        Args:
            - chunk_size: Max character limit of a chunk.
            - chunk_overlap: Max number of characters carried over from the
                             end of the last chunk to the next.
            - separators: Separators to split by, from coarsest to finest.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using given separators. Splits on the
        first separator, buffers the splits that fit and merges the buffer
        into chunks when an oversized split appears. Oversized splits are
        split until there are no remaining separators, and are sliced every
        chunk_size characters if there are no more separators left.

        Args:
            - text: The text to split.
            - separators: Separators to split by, from coarsest to finest.

        Returns:
            The chunks in order within chunk size.
        """
        # Return text if it already fits into chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Slice by char count if no more separators
        if not separators:
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        current_sep, next_sep = separators[0], separators[1:]
        splits = text.split(current_sep) if current_sep else list(text)

        final_chunks: List[str] = []
        unmerged_splits: List[str] = []

        # Append separator so it isn't lost
        for idx, split in enumerate(splits):
            if idx > 0 and current_sep:
                split = current_sep + split
            if len(split) <= self.chunk_size:
                unmerged_splits.append(split)
            else:
                if unmerged_splits:
                    final_chunks.extend(self.merge_splits(unmerged_splits))
                    unmerged_splits = []
                final_chunks.extend(self._split_text(split, next_sep))

        if unmerged_splits:
            final_chunks.extend(self.merge_splits(unmerged_splits))
        return final_chunks

    def merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge splits into chunks within chunk size accounting for overlap.
        Splits are added to the current chunk until the next one would overflow
        it. The chunk is then finalized, and pieces are dropped from its front
        until at most chunk_overlap characters remain. The remaining pieces
        start the next chunk, which creates the overlap.

        Args:
            - splits: A list of splits smaller than chunk size.

        Returns:
            The merged chunks in order. Consecutive chunks may share pieces or
            nothing if every piece is longer than chunk overlap.
        """
        chunks: List[str] = []
        current_chunk: List[str] = []
        total = 0

        for split in splits:
            split_len = len(split)
            if total + split_len > self.chunk_size and current_chunk:
                chunks.append("".join(current_chunk))
                while current_chunk and (total > self.chunk_overlap or
                                         total + split_len > self.chunk_size):
                    total -= len(current_chunk.pop(0))
            current_chunk.append(split)
            total += split_len

        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Public entry point to split text using chosen separators.

        Args:
            - text: The text to split.

        Returns:
            The chunks in order within chunk size.
        """
        return self._split_text(text, self.separators)


# Regex lookahead to ensure keywords stay in chunk during split
_PYTHON_SECTIONS = re.compile(r"(?=\n(?:class | (?:async )?def |     def ))")


def chunk_python(content: str, file_path: str,
                 chunk_size: int = 1000) -> List[Chunk]:
    """
    Chunks Python files into code blocks. Splits code based on regex
    keywords first, and falls back to RecursiveCharacterTextSplitter
    if chunk exceeds chunk size.

    Args:
        - content: The Python source code to be chunked.
        - file_path: The source file path.
        - chunk_size: Max char length of a chunk. 1000 by default.

    Returns:
        A list of Chunk objects containing Python code, file paths and
        character offsets.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(400, chunk_size // 2))

    splits: List[str] = []
    for section in _PYTHON_SECTIONS.split(content):
        if section:
            splits.extend(splitter.split_text(section))

    return _to_chunks(content, file_path, splitter, splits)


# Regex lookahead to ensure keywords stay in chunk
_MD_SECTIONS = re.compile(r"(?=\n#{1,6}[ \t])")

# Regex to parse a single header line into two capture groups
# Group 1: Captures the raw # string (e.g., ###) to measure its depth.
# Group 2: Captures the clean heading text (e.g., Installation).
_MD_HEADER = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
#                           └─Group 1─┘   └─Group 2─┘


def chunk_markdown(content: str, file_path: str,
                   chunk_size: int = 1200) -> List[Chunk]:
    """
    Chunks markdown along header boundaries using regex and attaches a
    trail for BM25 search (e.g., 'Parent > Child'). Falls back to
    RecursiveCharacterTextSplitter if chunk exceeds chunk size.

    Args:
        - content: The raw text of the Markdown file.
        - file_path: The source file path.
        - chunk size: Max char length of a chunk. 1200 by default.

    Returns:
        A list of Chunk objects containing text, file paths,
        character offsets and BM25 search prefixes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(300, chunk_size // 3)
    )

    sections = _MD_SECTIONS.split(content)
    # Strip whitespace and leading newlines
    sections = [s.lstrip("\n") for s in sections if s.strip()]
    header_stack: List[Tuple[int, str]] = []
    splits: List[str] = []
    bm25_texts: List[str] = []

    # Uses header stack to build a trail for BM25
    for section in sections:
        match = _MD_HEADER.match(section.split("\n", 1)[0])
        if match:
            depth = len(match.group(1))
            while header_stack and header_stack[-1][0] >= depth:
                header_stack.pop()
            header_stack.append((depth, match.group(2).strip()))

        trail = " > ".join(t for _, t in header_stack)
        prefix = f"{trail}: " if trail else ""

        # Fallback to RecursiveCharacterTextSplitter
        for split in splitter.merge_splits(splitter.split_text(section)):
            splits.append(split)
            bm25_texts.append(prefix + split)

    return _to_chunks(content, file_path, splitter, splits, bm25_texts)


def chunk_generic(content: str, file_path: str,
                  chunk_size: int = 1000) -> List[Chunk]:
    """
    Chunks plain text or unhandled file types using
    RecursiveCharacterTextSplitter by default.

    Args:
        - content: The raw text.
        - file_path: The source file path.
        - chunk size: Max char length of a chunk. 1000 by default.

    Returns:
        A list of Chunk objects containing text, file paths and
        character offsets.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(250, chunk_size // 4),
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    splits = splitter.split_text(content)
    return _to_chunks(content, file_path, splitter, splits)


def _to_chunks(content: str, file_path: str,
               splitter: RecursiveCharacterTextSplitter,
               splits: List[str],
               bm25_texts: Optional[List[str]] = None) -> List[Chunk]:
    """
    Converts raw text splits into Chunk objects with exact character spans.
    Iterates through splits with a cursor to keep track of first and last
    character index offsets.

    Args:
        - content: The full text content of the source file.
        - file_path: The source file path.
        - splitter: The RecursiveCharacterTextSplitter instance
                    used to generate splits.
        - splits: A list of raw text slices representing chunk contents.
        - bm25_texts: An optional parallel list of trail text used for
                    search indexing. None by default.

    Returns:
        A list of Chunk objects containing text, file paths,
        character offsets and optional BM25 search trails.
    """
    chunks: List[Chunk] = []
    cursor = 0

    for idx, split in enumerate(splits):
        start_char = content.find(split, cursor)
        if start_char == -1:
            start_char = cursor
        end_char = start_char + len(split)

        chunks.append(
            Chunk(
                file_path=file_path,
                content=split,
                first_character_index=start_char,
                last_character_index=end_char,
                bm25_text=bm25_texts[idx] if bm25_texts is not None else None
            )
        )
        cursor = max(end_char - splitter.chunk_overlap, start_char + 1)

    return chunks
