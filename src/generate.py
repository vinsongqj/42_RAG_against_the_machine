import json
import re
import textwrap
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from src.models import MinimalSource


OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:0.6b"
_REQUEST_TIMEOUT_SECS = 120
_ENUMERATION_QUERY = re.compile(
    r"""
    \b(what|which)\b
    [^?.]*
    \b(models?|formats?|platforms?|backends?|versions?|options?|types?)\b
    [^?.]*
    \b(support|supports|supported|available)\b
    | \blist\b
    | \benumerate\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _is_enumeration_query(question: str) -> bool:
    return bool(_ENUMERATION_QUERY.search(question))


def _call_ollama(prompt: str, max_new_tokens: int) -> str:
    config: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,  # Waits for complete response before returning
        "think": False,  # Turn off reasoning tokens and return final answer
        "keep_alive": "30m",  # How long to keep the model in memory
        "options": {
            "temperature": 0.1,  # Reduces creative or random responses
            "top_p": 0.8,  # Considers 80% of likely tokens (recommended)
            "num_predict": max_new_tokens,  # Max tokens for response
            "num_ctx": 4096  # Context window allocated in memory
        }
    }
    # Creates a POST request for Ollama's API containing config
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=json.dumps(config).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        # Sends a POST request to Ollama with a timeout limit
        with urllib.request.urlopen(request,
                                    timeout=_REQUEST_TIMEOUT_SECS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not get a response from Ollama at {OLLAMA_HOST} ({e}), "
            f"Try running `ollama serve`, then `ollama pull {DEFAULT_MODEL}`"
        ) from e
    except TimeoutError as e:
        raise RuntimeError(
            f"Ollama did not respond within {_REQUEST_TIMEOUT_SECS}s."
        ) from e
    # Extracts output from JSON response and trims whitespace
    return str(body.get("response", "")).strip()


def select_sources(sources: List[MinimalSource], top_k: int,
                   min_chars: int = 150) -> List[MinimalSource]:
    # Filters out sources shorter than min chars unless all sources are small
    kept = [s for s in sources
            if (s.last_character_index - s.first_character_index) >= min_chars
            ] or sources
    docs = [s for s in kept if s.file_path.endswith(".md")]
    code = [s for s in kept if not s.file_path.endswith(".md")]
    # At least 2 .md files
    if len(docs) >= 2:
        return docs[:top_k]
    return (docs + code)[:top_k]


def _get_context_spans(sources: List[MinimalSource],
                       padding: int) -> List[Tuple[str, int, int]]:
    spans_per_file: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for rank, src in enumerate(sources):
        start = max(0, src.first_character_index - padding)
        end = src.last_character_index + padding
        spans_per_file[src.file_path].append((start, end, rank))
    result: List[Tuple[str, int, int, int]] = []
    for file_path, spans in spans_per_file.items():
        spans.sort(key=lambda w: w[0])
        cur_start, cur_end, cur_rank = spans[0]
        for start, end, rank in spans[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                cur_rank = min(cur_rank, rank)
            else:
                result.append((file_path, cur_start, cur_end, cur_rank))
                cur_start, cur_end, cur_rank = start, end, rank
        result.append((file_path, cur_start, cur_end, cur_rank))
    result.sort(key=lambda m: m[3])
    return [(file_path, start, end) for file_path, start, end, _ in result]


def generate_answer(question: str, sources: List[MinimalSource],
                    max_new_tokens: int = 256, top_k: int = 6,
                    padding: int = 1500,
                    enum_padding: int = 4000,
                    max_span_chars: int = 3000) -> str:
    sources = select_sources(sources, top_k)
    if not sources:
        return "No relevant sources retrieved."
    is_enumeration = _is_enumeration_query(question)
    max_context_chars = 12000
    context_parts: List[str] = []
    total_chars = 0
    file_cache: Dict[str, Optional[str]] = {}
    chosen_padding = (enum_padding if is_enumeration
                      else padding)
    spans = _get_context_spans(sources, chosen_padding)

    for file_path, start, end in spans:
        if file_path not in file_cache:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_cache[file_path] = f.read()
            except Exception:
                print(f"Warning: could not read {file_path}; skipping...")
                file_cache[file_path] = None
        content = file_cache[file_path]
        if content is None or start >= len(content):
            continue
        snippet = content[start:min(end, start + max_span_chars)]
        part = f"File: {file_path}\n```\n{snippet}\n```"

        if not context_parts and len(part) > max_context_chars:
            part = part[:max_context_chars] + "\n...[truncated]"
        elif context_parts and total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        total_chars += len(part)

    if not context_parts:
        return "No readable sources to answer from."

    if len(context_parts) < len(spans):
        print(f"Using top {len(context_parts)}/{len(spans)} spans "
              f"({total_chars} chars) to stay within the context budget.")
    context = "\n\n".join(context_parts)

    if is_enumeration:
        answer_style_rule = (
            "- This question asks for a list: enumerate every specific "
            "item named in the documents "
            "(e.g. model names, formats, platforms) "
            "rather than summarizing them as a category."
        )
    else:
        answer_style_rule = (
            "- Explain the key idea directly in 2-4 sentences, "
            "using the precise terms and concepts the documents use "
            "rather than a generic paraphrase."
        )

    prompt = textwrap.dedent(f"""\
    You are a technical assistant for the vLLM codebase and documentation.
    Answer the question using only the documents below.

    Rules:
    {answer_style_rule}
    - If the documents contain commands, code, or steps, include them.
    - Never mention "the context", "the documents", or "the provided text".
    - Do not add facts that are not in the documents.
    - If the documents do not answer the question, say only: \
    "I could not find this in the documentation."

    Documents:
    {context}

    Question: {question}

    Answer:""")

    return _call_ollama(prompt, max_new_tokens)
