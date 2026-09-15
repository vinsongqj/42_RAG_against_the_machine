import re
from typing import List, Optional, Tuple
from src.models import CodeChunk


_CAMEL_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_LOWER = re.compile(r"([a-z0-9])([A-Z])")


def code_friendly_text(text: str) -> str:
    """Split camelCase / PascalCase / snake_case into space-separated tokens."""
    text = _CAMEL_ACRONYM.sub(r"\1 \2", text)
    text = _CAMEL_LOWER.sub(r"\1 \2", text)
    return text.replace("_", " ")


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


_HEADER_LINE_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")

_MD_HEADER_BOUNDARY_RE = re.compile(r"(?=\n#{1,6}[ \t])")


def chunk_markdown(content: str, file_path: str, chunk_size: int = 1200) -> List[CodeChunk]:
 
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(300, chunk_size // 3),
        separators=["\n\n", "\n", " ", ""],
    )

    sections = _MD_HEADER_BOUNDARY_RE.split(content)
    sections = [s.lstrip("\n") for s in sections if s.strip()]

    header_stack: List[Tuple[int, str]] = []
    pieces: List[str] = []
    bm25_texts: List[str] = []

    for section in sections:
        first_line = section.split("\n", 1)[0]
        m = _HEADER_LINE_RE.match(first_line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))

        trail = " > ".join(t for _, t in header_stack)
        prefix = f"{trail}: " if trail else ""

        if len(section) <= chunk_size:
            pieces.append(section)
            bm25_texts.append(prefix + section)
        else:
            sub_merged = fallback_splitter.merge_pieces(
                fallback_splitter.split_text(section)
            )
            for sub in sub_merged:
                pieces.append(sub)
                bm25_texts.append(prefix + sub)

    return _to_chunks(
        content, file_path, fallback_splitter,
        texts=pieces, bm25_texts=bm25_texts,
    )


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
