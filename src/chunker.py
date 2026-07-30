from typing import List
from src.models import CodeChunk


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200,
                 separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        if len(text) <= self.chunk_size:
            return [text]

        # If no separators left, split by characters
        if not separators:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        current_sep = separators[0]
        next_sep = separators[1:]   # can be empty

        splits = text.split(current_sep) if current_sep else list(text)
        good_splits = []

        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, current_sep))
                    good_splits = []
                # Recurse with the remaining separators
                recursive_splits = self._split_text(s, next_sep)
                final_chunks.extend(recursive_splits)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, current_sep))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        separator_len = len(separator)

        for d in splits:
            len_d = len(d)
            if total + len_d + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    docs.append(separator.join(current_doc))
                    # Apply overlap: remove from front until within overlap limit
                    while current_doc and total > self.chunk_overlap:
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

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self.separators)


def chunk_python(content: str, file_path: str, chunk_size: int = 1500) -> List[CodeChunk]:
    python_separators = ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=300,
        separators=python_separators
    )
    return _to_chunks(content, file_path, splitter)


def chunk_markdown(content: str, file_path: str, chunk_size: int = 2000) -> List[CodeChunk]:
    markdown_separators = ["\n#{1,6} ", "\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=markdown_separators
    )
    return _to_chunks(content, file_path, splitter)


def _to_chunks(content: str, file_path: str, splitter: RecursiveCharacterTextSplitter) -> List[CodeChunk]:
    texts = splitter.split_text(content)
    chunks = []
    cursor = 0

    for text in texts:
        start_char = content.find(text, cursor)
        # fallback if not found
        if start_char == -1:
            start_char = cursor
        end_char = start_char + len(text)
        # safety truncate (should not happen)
        if end_char - start_char > splitter.chunk_size:
            end_char = start_char + splitter.chunk_size
            text = content[start_char:end_char]

        chunks.append(
            CodeChunk(
                file_path=file_path,
                content=text,
                first_character_index=start_char,
                last_character_index=end_char
            )
        )
        # **** THIS IS THE KEY OVERLAP ****
        cursor = (end_char - splitter.chunk_overlap
                  if splitter.chunk_overlap < len(text)
                  else start_char + 1)

    return chunks