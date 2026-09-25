import json
import re
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from src.models import MinimalSource

OLLAMA_HOST = "http://localhost:11434"
MODEL_ID = "qwen3:0.6b"
_KEEP_ALIVE = "30m"
_REQUEST_TIMEOUT_SECONDS = 120

_FILLER_PREFIX_RE = re.compile(
    r"^(the answer is:?|based on the (provided )?context,?|"
    r"according to the context,?|answer:?)\s*",
    re.IGNORECASE,
)

_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s)]+")

_ENUMERATION_QUERY_RE = re.compile(
    r"\b(what|which)\b[^?.]*\b(models?|formats?|platforms?|backends?|versions?|options?|types?)\b"
    r"[^?.]*\b(support|supports|supported|available)\b"
    r"|\blist\b|\benumerate\b",
    re.IGNORECASE,
)

_META_COMMENTARY_RE = re.compile(
    r"\bprovided (document|documents|context|text)\b"
    r"|\bnot (explicitly )?(mentioned|specified|provided)\b.{0,25}\b(document|documents|context)s?\b"
    r"|\bbased on\b.{0,40}\b(information|context)\b"
    r"|\bthe answer would be\b"
    r"|\bthe information given\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[a-z0-9)"\'])([.!?]+)\s+(?=[A-Z"])')


def _strip_filler_prefix(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _FILLER_PREFIX_RE.sub("", text, count=1).lstrip()
    return text


def _ground_urls(answer: str, context: str) -> str:
    def _replace_markdown_link(match: "re.Match[str]") -> str:
        label, url = match.group(1), match.group(2)
        return match.group(0) if url in context else label

    def _replace_bare_url(match: "re.Match[str]") -> str:
        url = match.group(0)
        return url if url in context else ""

    answer = _MD_LINK_RE.sub(_replace_markdown_link, answer)
    answer = _BARE_URL_RE.sub(_replace_bare_url, answer)
    return re.sub(r"[ \t]{2,}", " ", answer)


def _is_enumeration_query(question: str) -> bool:
    return bool(_ENUMERATION_QUERY_RE.search(question))


def _strip_meta_commentary(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if not line.strip():
            cleaned_lines.append(line)
            continue
        marked = _SENTENCE_SPLIT_RE.sub(lambda m: m.group(1) + "\x00", line)
        sentences = marked.split("\x00")
        kept = [s for s in sentences if not _META_COMMENTARY_RE.search(s)]
        cleaned_line = " ".join(s.strip() for s in kept if s.strip())
        cleaned_lines.append(cleaned_line if cleaned_line else line)
    cleaned = "\n".join(cleaned_lines).strip(" \n\"")
    return cleaned if cleaned.strip() else text


def _call_ollama(prompt: str, max_new_tokens: int) -> str:
    payload: Dict[str, Any] = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": _KEEP_ALIVE,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "num_predict": max_new_tokens,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. Is `ollama serve` "
            f"running, and has `ollama pull {MODEL_ID}` been run? ({e})"
        ) from e
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Ollama returned HTTP {e.code} for model {MODEL_ID!r}: {detail}. "
            f"If the model isn't pulled yet, run `ollama pull {MODEL_ID}`."
        ) from e

    return _strip_filler_prefix(str(body.get("response", "")).strip())


def select_sources(sources, top_k, min_chars=150):
    size = lambda s: s.last_character_index - s.first_character_index
    kept = [s for s in sources if size(s) >= min_chars] or sources
    docs = [s for s in kept if s.file_path.endswith(".md")]
    code = [s for s in kept if not s.file_path.endswith(".md")]
    if len(docs) >= 2:
        return docs[:top_k]
    return (docs + code)[:top_k]


def _expand_and_merge_windows(
    sources: List[MinimalSource], window_chars: int,
) -> List[Tuple[str, int, int]]:
    windows_by_file: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for rank, src in enumerate(sources):
        start = max(0, src.first_character_index - window_chars)
        end = src.last_character_index + window_chars
        windows_by_file[src.file_path].append((start, end, rank))

    merged: List[Tuple[str, int, int, int]] = []
    for file_path, windows in windows_by_file.items():
        windows.sort(key=lambda w: w[0])
        cur_start, cur_end, cur_rank = windows[0]
        for start, end, rank in windows[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                cur_rank = min(cur_rank, rank)
            else:
                merged.append((file_path, cur_start, cur_end, cur_rank))
                cur_start, cur_end, cur_rank = start, end, rank
        merged.append((file_path, cur_start, cur_end, cur_rank))

    merged.sort(key=lambda m: m[3])
    return [(file_path, start, end) for file_path, start, end, _ in merged]


def generate_answer(question: str, sources: List[MinimalSource], max_new_tokens: int = 256, top_k: int = 6,
                    context_window_chars: int = 1500,
                    enumeration_context_window_chars: int = 4000,
                    max_span_chars: int = 3000) -> str:

    sources = select_sources(sources, top_k)
    if not sources:
        return "No relevant sources retrieved."

    is_enumeration = _is_enumeration_query(question)

    max_context_chars = 12000
    context_parts: List[str] = []
    total_chars = 0
    file_cache: Dict[str, str] = {}

    effective_window_chars = enumeration_context_window_chars if is_enumeration else context_window_chars
    spans = _expand_and_merge_windows(sources, effective_window_chars)

    for file_path, start, end in spans:
        if file_path not in file_cache:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_cache[file_path] = f.read()
            except Exception:
                file_cache[file_path] = f"[Could not read file: {file_path}]"
        content = file_cache[file_path]

        end = min(end, start + max_span_chars)
        snippet = content[start:min(end, len(content))]
        part = f"File: {file_path}\n```\n{snippet}\n```"

        if not context_parts and len(part) > max_context_chars:
            part = part[:max_context_chars] + "\n...[truncated]"
        elif context_parts and total_chars + len(part) > max_context_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    if len(context_parts) < len(spans):
        print(f"Using top {len(context_parts)}/{len(spans)} spans "
              f"({total_chars} chars) to stay within the context budget.")

    context = "\n\n".join(context_parts)

    if is_enumeration:
        answer_style_rule = (
            "- This question asks for a list: enumerate every specific item named in the "
            "documents (e.g. model names, formats, platforms) rather than summarizing them as a category."
        )
    else:
        answer_style_rule = (
            "- Explain the key idea directly in 2-4 sentences, using the precise terms and "
            "concepts the documents use rather than a generic paraphrase."
        )

    prompt = f"""You are a technical assistant for the vLLM codebase and documentation.
Answer the question using only the documents below.

Rules:
{answer_style_rule}
- If the documents contain commands, code, or steps, include them.
- Never mention "the context", "the documents", or "the provided text".
- Do not add facts that are not in the documents.
- If the documents do not answer the question, say only: "I could not find this in the documentation."

Documents:
{context}

Question: {question}

Answer:"""

    answer = _call_ollama(prompt, max_new_tokens)
    answer = _ground_urls(answer, context)
    return _strip_meta_commentary(answer)
