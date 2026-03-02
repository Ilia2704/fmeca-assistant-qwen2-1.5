from __future__ import annotations

import logging
import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("local_llm")

MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")

_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForCausalLM | None = None


def load_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load HF tokenizer+model once and cache them."""
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    log.info("Loading local model: %s", MODEL_NAME)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    _TOKENIZER, _MODEL = tokenizer, model
    return tokenizer, model


def get_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return cached model, loading it on first use."""
    if _TOKENIZER is None or _MODEL is None:
        return load_model()
    return _TOKENIZER, _MODEL


def generate_answer_en(
    question_en: str,
    context: str = "",
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Generate an English answer using the local model.

    This is intentionally simple and deterministic (temperature≈0)
    to make latency profiling easier.
    """
    tokenizer, model = get_model()

    system_prompt = (
        "You are a helpful assistant working with an FMECA "
        "(Failure Modes, Effects and Criticality Analysis) knowledge base. "
        "Use the provided context when available; if it is empty, "
        "answer based on your general knowledge."
    )

    parts: list[str] = []
    if context:
        parts.append("Context:\n" + context.strip())
    parts.append("Question:\n" + question_en.strip())
    user_prompt = "\n\n".join(parts)

    # Very simple chat formatting that works fine for Qwen-style instruct models.
    text = f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature if temperature > 0.0 else 0.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    gen_ids = output[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    log.info("Local LLM generated answer_en_len=%d", len(answer))
    return answer