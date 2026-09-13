"""Answer generation via a local Ollama daemon serving Qwen.

Previously this module loaded Qwen3-0.6B weights with `transformers` on
every cold start -- and since each CLI invocation (`answer`, `search_dataset`
+ answer_dataset, the API's first request, ...) is its own process, that
meant re-loading the weights from disk on every single call.

Ollama runs the model once, as a background daemon, and keeps it resident
in memory between requests (see `keep_alive` below). This module no longer
loads anything itself -- it just sends an HTTP request to that already
running server. One-time setup on the machine running this code:

    ollama pull qwen3:0.6b
    ollama serve        # usually already running as a background service

`urllib` (stdlib) is used instead of `requests`/`httpx` so this doesn't add
a new dependency on top of what the bonuses already need.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from src.models import MinimalSource

OLLAMA_HOST = "http://localhost:11434"
MODEL_ID = "qwen3:0.6b"

# How long Ollama keeps the model loaded in memory after the last request.
# Ollama's own default is "5m"; a batch run (search_dataset/answer_dataset,
# or many API calls in a row) can easily go idle longer than that between
# some questions, so we ask it to stay resident for the whole session.
_KEEP_ALIVE = "30m"
_REQUEST_TIMEOUT_SECONDS = 120


def _call_ollama(prompt: str, max_new_tokens: int) -> str:
    payload: Dict[str, Any] = {
        "model": MODEL_ID,
        "prompt": prompt,
        "stream": False,
        "think": False,  # Qwen3 thinks by default; we just want the answer.
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

    return str(body.get("response", "")).strip()


def generate_answer(
    question: str,
    sources: List[MinimalSource],
    max_new_tokens: int = 256,
) -> str:
    """
    Generate an answer grounded in the provided sources, via a local Ollama
    daemon running Qwen3-0.6B.
    Sources must contain file_path and character range; we read the actual
    content from the original files to include in the prompt.
    """
    if not sources:
        return "No relevant sources retrieved."

    # qwen3:0.6b's real context window is 32K+ tokens; budgeting ~3000
    # *characters* (well under 1000 tokens) was needlessly conservative and
    # meant only the first 2-4 retrieved sources ever reached the model --
    # everything ranked below that, including cases where the actual answer
    # lived in source #5-#10, was silently discarded.
    max_context_chars = 12000

    # Add whole sources, best-ranked first, until the budget is used. This
    # never truncates a source mid-file/mid-code-block; if there isn't room
    # for everything, it's the lowest-ranked *whole* sources that get
    # dropped, not an arbitrary character offset partway through some
    # source in the middle of the ranking (which is what joining every
    # source first and then slicing the combined string used to do).
    context_parts: List[str] = []
    total_chars = 0
    for src in sources:
        try:
            with open(src.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = f"[Could not read file: {src.file_path}]"
        # Slice the exact span
        snippet = content[src.first_character_index:src.last_character_index]
        part = f"File: {src.file_path}\n```\n{snippet}\n```"

        if not context_parts and len(part) > max_context_chars:
            # Even the single top-ranked source alone exceeds the budget.
            # Some context beats none, so include a truncated version of
            # just this one rather than dropping it entirely.
            part = part[:max_context_chars] + "\n...[truncated]"
        elif context_parts and total_chars + len(part) > max_context_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    if len(context_parts) < len(sources):
        print(f"Using top {len(context_parts)}/{len(sources)} sources "
              f"({total_chars} chars) to stay within the context budget.")

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""
You are a precise technical assistant for software codebases.
Answer the question based solely on the provided context.
If the context does not contain the answer, say so.

Context:
{context}

Question: {question}

Answer:
"""
    return _call_ollama(prompt, max_new_tokens)
