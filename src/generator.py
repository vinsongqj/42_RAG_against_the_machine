import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from src.models import MinimalSource

MODEL_ID = "Qwen/Qwen3-0.6B"
tokenizer = None
model = None


def _load_model():
    global tokenizer, model
    if tokenizer is None:
        print(f"Loading model {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Optimize for CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,  # float32 is faster on CPU than float16
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Optimize for CPU inference
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True  # Enable KV cache
        
        print("Model loaded!")
    return tokenizer, model


def generate_answer(question: str,
                    sources: List[MinimalSource],
                    max_new_tokens: int = 128) -> str:
    """
    Generate an answer grounded in the provided sources using the Qwen model.
    Sources must contain file_path and character range; we read the actual content
    from the original files to include in the prompt.
    """
    if not sources:
        return "No relevant sources retrieved."

    # Build context from sources (read the actual file content)
    context_parts = []
    for src in sources:
        try:
            with open(src.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = f"[Could not read file: {src.file_path}]"
        # Slice the exact span
        snippet = content[src.first_character_index:src.last_character_index]
        context_parts.append(f"File: {src.file_path}\n```\n{snippet}\n```")

    context = "\n\n".join(context_parts)

    # Truncate context if too long (rough token estimation)
    # Approx 4 chars per token, we leave room for system+user prompt and answer
    max_context_chars = 3000  # adjust based on model's context window
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n...[truncated]"

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
    tokenizer, model = _load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.8,
            do_sample=True,
        )
    # Decode only new tokens
    output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return answer.strip()