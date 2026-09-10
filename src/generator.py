import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, List, Optional, Tuple
from src.models import MinimalSource

MODEL_ID = "Qwen/Qwen3-0.6B"
tokenizer: Optional[Any] = None
model: Optional[Any] = None


def _load_model():
    global tokenizer, model
    if tokenizer is None:
        print(f"Loading model {MODEL_ID}...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        print("Model loaded!", file=sys.stderr)
    return tokenizer, model


def generate_answer(
    question: str,
    sources: List[MinimalSource],
    max_new_tokens: int = 200,
) -> str:
    """Generate an answer grounded in the provided sources using Qwen."""
    if not sources:
        return "No relevant sources retrieved."
    
    context_parts = []
    for src in sources:
        try:
            with open(src.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = f"[Could not read file: {src.file_path}]"
        snippet = content[src.first_character_index:src.last_character_index]
        snippet = snippet.replace("```", "")
        context_parts.append(f"File: {src.file_path}\n{snippet}")

    context = "\n\n".join(context_parts)
    if len(context) > 6000:
        context = context[:6000] + "\n...[truncated]"

    system_msg = (
        "You are a documentation assistant. Answer the user's question in "
        "1-3 sentences of plain prose, using only the provided context. "
        "Do not ask questions. Do not offer options. Do not use lists, "
        "Markdown, or code fences. If the context does not contain the "
        "answer, reply exactly: 'The context does not contain the answer.'"
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    tokenizer, model = _load_model()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # Trim any trailing reasoning/follow-up the model tacked on.
    for marker in ("\n\nQuestion:", "\n\nOptions:", "\nAnswer:", "\n```"):
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    answer = answer.replace("```", "").strip()

    return answer or "No answer generated."