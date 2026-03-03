from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from langgraph_pipeline import PipelineState, run_pipeline
from local_llm import get_model, load_model
from monitoring.prometheus_metrics import track_request

log = logging.getLogger("chat_core")


def build_request_id(user_text: str) -> str:
    """Return a short request id (first 10 characters of the user query)."""
    text = (user_text or "").strip()
    if not text:
        return "empty_query"
    return text[:10]


def load() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Keep old public API: return (tokenizer, model)."""
    tokenizer, model = load_model()
    return tokenizer, model


def init_history() -> List[Dict[str, Any]]:
    """Very simple chat-history placeholder kept for compatibility."""
    return []


def generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    history: List[Dict[str, Any]],
    user_text: str,
    *,
    do_sample: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    """Main entry point used by run_local.py / Streamlit.

    All orchestration (lang detect → translate RU→EN → retrieval → local LLM → EN→RU)
    is delegated to LangGraph. Local LLM is called inside langgraph_pipeline via
    local_llm.generate_answer_en().
    """
    # Ensure model is loaded (no-op if already done)
    _ = get_model()

    # Build short request id from the user query (first 10 characters)
    request_id = build_request_id(user_text)
    log.info("generate() started, request_id=%s", request_id)

    # Track latency / in-flight metrics in Prometheus with per-request label
    with track_request(request_id):
        state_out: PipelineState = run_pipeline(
            user_query=user_text,
            tokenizer=tokenizer,
            model=model,
            history=history,
            request_id=request_id,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    answer = state_out.get("answer_ru") or state_out.get("answer_en") or ""
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})

    log.info(
        "generate() completed, request_id=%s, answer_len=%d",
        request_id,
        len(answer),
    )
    return answer