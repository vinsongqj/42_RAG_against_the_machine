import re
from typing import List, Optional, Tuple
from src.models import CodeChunk


# ============================================================================
# Identifier-aware text normalisation (used by BM25 at index & query time)
# ============================================================================

_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_LOWER = re.compile(r"([a-z0-9])([A-Z])")


def code_friendly_text(text: str) -> str:
    """Split camelCase / PascalCase / snake_case into space-separated tokens."""
    text = _CAMEL_ACRONYM.sub(r"\1 \2", text)
    text = _CAMEL_LOWER.sub(r"\1 \2", text)
    return text.replace("_", " ")


# ============================================================================
# Recursive splitter
# ============================================================================

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        current_sep = separators[0]
        next_sep = separators[1:]

        splits = text.split(current_sep) if current_sep else list(text)
        good_splits: List[str] = []

        for i, s in enumerate(splits):
            if i > 0 and current_sep:
                s = current_sep + s

            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, ""))
                    good_splits = []
                final_chunks.extend(self._split_text(s, next_sep))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, ""))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0
        separator_len = len(separator)

        for d in splits:
            len_d = len(d)
            if total + len_d + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    docs.append(separator.join(current_doc))
                    while current_doc and (
                        total > self.chunk_overlap
                        or total + len_d + separator_len > self.chunk_size
                    ):
                        removed = current_doc.pop(0)
                        total -= len(removed) + separator_len
                current_doc.append(d)
                total += len_d + (separator_len if len(current_doc) > 1 else 0)
            else:
                current_doc.append(d)
                total += len_d + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            docs.append(separator.join(current_doc))
        return docs

    def merge_pieces(self, pieces: List[str], separator: str = "") -> List[str]:
        return self._merge_splits(pieces, separator)

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)


# ============================================================================
# Python chunking
# ============================================================================

_PY_BOUNDARY_RE = re.compile(r"(?=\n(?:class |(?:async )?def |    def ))")


def chunk_python(content: str, file_path: str, chunk_size: int = 1000) -> List[CodeChunk]:
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(400, chunk_size // 2),
        separators=["\n\n", "\n", " ", ""],
    )

    sections = _PY_BOUNDARY_RE.split(content)

    pieces: List[str] = []
    for section in sections:
        if not section:
            continue
        if len(section) <= chunk_size:
            pieces.append(section)
        else:
            pieces.extend(fallback.split_text(section))

    return _to_chunks(content, file_path, fallback, texts=pieces)


# ============================================================================
# Markdown chunking
# ============================================================================

# ATX header line: 1-6 #, then whitespace, then title (trailing #'s optional).
_HEADER_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")

# Zero-width lookahead: split *before* the newline that precedes a header.
# Keeps the "\n" and "#"s attached to the following piece (no text lost).
# Accepts any whitespace after the #'s, so "#Header" with a tab also matches.
_MD_HEADER_BOUNDARY_RE = re.compile(r"(?=\n#{1,6}[ \t])")


def chunk_markdown(content: str, file_path: str, chunk_size: int = 1200) -> List[CodeChunk]:
    """Split Markdown on header boundaries, then merge undersized sections forward.

    Two properties matter for recall:

    * Sections are split at every header boundary (not merged wholesale the
      way the very first version did, which diluted IoU with short spans).

    * A section whose own body is short is then merged forward into the
      next section.  A "# Supported Models" header followed by two lines of
      prose and then a child section with the actual model table was
      previously emitted as a ~280-char stub that BM25 could not score
      above unrelated files; merging it into the child section gives one
      chunk with both the intro *and* the list.

    * The header trail ("Guide > Installation > macOS") is prepended to
      ``bm25_text`` for BM25 only.  Stored content and character offsets
      remain exactly what is on disk, so ``generate_answer`` still slices
      the raw file correctly.
    """
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(300, chunk_size // 3),
        separators=["\n\n", "\n", " ", ""],
    )

    # Raw sections are contiguous substrings of `content` (the lookahead
    # split consumes nothing), so concatenating adjacent sections still
    # reproduces the original text — required for _to_chunks() to locate
    # merged chunks via content.find().  Do NOT lstrip here.
    raw_sections = _MD_HEADER_BOUNDARY_RE.split(content)
    raw_sections = [s for s in raw_sections if s.strip()]

    # Pass 1: tag each section with its header level and cumulative trail.
    header_stack: List[Tuple[int, str]] = []
    tagged: List[Tuple[int, str, str, str]] = []  # level, title, text, trail
    for section in raw_sections:
        # Sections may begin with a "\n" (kept intact above); strip it for
        # header detection only.
        first_line = section.lstrip("\n").split("\n", 1)[0]
        m = _HEADER_LINE_RE.match(first_line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))
        else:
            # Non-header section inherits the current trail; its "level" is
            # the depth of the open header, for the merge check below.
            level = header_stack[-1][0] if header_stack else 0
            title = ""
        trail = " > ".join(t for _, t in header_stack)
        tagged.append((level, title, section, trail))

    # Pass 2: greedy merge.  A section that would otherwise produce an
    # undersized chunk absorbs the following section.  Only undersized
    # sections trigger this, so large sections stay separate and the
    # earlier fix (no merging across headers) remains in effect for them.
    MIN_SECTION_CHARS = 500
    merged: List[Tuple[int, str, str, str]] = []
    buffer: Optional[Tuple[int, str, str, str]] = None
    for level, title, text, trail in tagged:
        if buffer is None:
            buffer = (level, title, text, trail)
        elif len(buffer[2]) < MIN_SECTION_CHARS:
            # Merge the new section into the buffer.  Use the newer trail
            # so bm25_text reflects the most specific header.
            buffer = (buffer[0], buffer[1], buffer[2] + text, trail)
        else:
            merged.append(buffer)
            buffer = (level, title, text, trail)
    if buffer is not None:
        merged.append(buffer)

    # Pass 3: emit pieces.  Anything larger than chunk_size is further
    # split by the fallback splitter; every piece gets the header-trail
    # prefix added to bm25_text only (not to stored content).
    pieces: List[str] = []
    bm25_texts: List[str] = []
    for _level, _title, text, trail in merged:
        prefix = f"{trail}: " if trail else ""
        if len(text) <= chunk_size:
            pieces.append(text)
            bm25_texts.append(prefix + text)
        else:
            sub_merged = fallback_splitter.merge_pieces(
                fallback_splitter.split_text(text)
            )
            for sub in sub_merged:
                pieces.append(sub)
                bm25_texts.append(prefix + sub)

    return _to_chunks(
        content, file_path, fallback_splitter,
        texts=pieces, bm25_texts=bm25_texts,
    )

# ============================================================================
# Shared helper
# ============================================================================

def _to_chunks(content: str, file_path: str,
               splitter: RecursiveCharacterTextSplitter,
               texts: Optional[List[str]] = None,
               bm25_texts: Optional[List[str]] = None) -> List[CodeChunk]:
    texts = texts if texts is not None else splitter.split_text(content)
    chunks: List[CodeChunk] = []
    cursor = 0

    for i, text in enumerate(texts):
        start_char = content.find(text, cursor)
        if start_char == -1:
            start_char = cursor
        end_char = start_char + len(text)
        if end_char - start_char > splitter.chunk_size:
            end_char = start_char + splitter.chunk_size
            text = content[start_char:end_char]

        bm25 = bm25_texts[i] if bm25_texts is not None else None

        chunks.append(
            CodeChunk(
                file_path=file_path,
                content=text,
                first_character_index=start_char,
                last_character_index=end_char,
                bm25_text=bm25,
            )
        )
        if splitter.chunk_overlap < len(text):
            cursor = max(end_char - splitter.chunk_overlap, start_char + 1)
        else:
            cursor = start_char + 1

    return chunks