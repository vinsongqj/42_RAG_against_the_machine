import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List
from src.models import MinimalSource

OLLAMA_HOST = "http://localhost:11434"
MODEL_ID = "qwen3:0.6b"

_FILLER_PREFIX_RE = re.compile(
    r"^(the answer is:?|based on the (provided )?context,?|"
    r"according to the context,?|answer:?)\s*",
    re.IGNORECASE,
)


def _strip_filler_prefix(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _FILLER_PREFIX_RE.sub("", text, count=1).lstrip()
    return text


_KEEP_ALIVE = "30m"
_REQUEST_TIMEOUT_SECONDS = 120


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


def generate_answer(
    question: str,
    sources: List[MinimalSource],
    max_new_tokens: int = 256,
) -> str:
    if not sources:
        return "No relevant sources retrieved."

    max_context_chars = 12000

    context_parts: List[str] = []
    total_chars = 0
    for src in sources:
        try:
            with open(src.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = f"[Could not read file: {src.file_path}]"

        snippet = content[src.first_character_index:src.last_character_index]
        part = f"File: {src.file_path}\n```\n{snippet}\n```"

        if not context_parts and len(part) > max_context_chars:
            part = part[:max_context_chars] + "\n...[truncated]"
        elif context_parts and total_chars + len(part) > max_context_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    if len(context_parts) < len(sources):
        print(f"Using top {len(context_parts)}/{len(sources)} sources "
              f"({total_chars} chars) to stay within the context budget.")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise technical assistant for software codebases.
Answer the question based solely on the context below.
If the context does not contain the answer, state that directly in one sentence.

Respond with only the substantive answer itself: no restating the question,
and no introductory phrase before the actual information.

Context:
{context}

Question: {question}

Answer:"""

    return _call_ollama(prompt, max_new_tokens)
