from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from langgraph_pipeline import PipelineState, run_pipeline
from local_llm import get_backend, get_model, load_model  # backend selector
from monitoring.prometheus_metrics import track_request

log = logging.getLogger("chat_core")


def build_request_id(user_text: str) -> str:
    """Return a short request id (first 10 characters of the user query)."""
    text = (user_text or "").strip()
    if not text:
        return "empty_query"
    return text[:10]


def load() -> Tuple[AutoTokenizer, Any]:
    """Keep old public API: return (tokenizer, model).

    Model can be HF AutoModelForCausalLM or vLLM.LLM instance.
    """
    tokenizer, model = load_model()
    return tokenizer, model


def init_history() -> List[Dict[str, Any]]:
    """Very simple chat-history placeholder kept for compatibility."""
    return []


def generate(
    tokenizer: AutoTokenizer,
    model: Any,
    history: List[Dict[str, Any]],
    user_text: str,
    *,
    scenario: str | None = None,
    do_sample: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    # Ensure model is loaded (no-op if already done)
    _ = get_model()

    # Build short request id from the user query (first 10 characters)
    request_id = build_request_id(user_text)
    backend = get_backend()  # "hf" or "vllm"
    scenario_label = scenario or "interactive"

    log.info(
        "generate() started, request_id=%s, backend=%s, scenario=%s",
        request_id,
        backend,
        scenario_label,
    )

    # Track latency / inflight and token metrics in Prometheus
    with track_request(request_id, backend=backend, scenario=scenario_label) as metrics_ctx:
        state_out: PipelineState = run_pipeline(
            user_query=user_text,
            tokenizer=tokenizer,
            model=model,
            history=history,
            request_id=request_id,
            backend=backend,
            scenario=scenario_label,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        metrics_ctx["prompt_tokens"] = int(state_out.get("prompt_tokens") or 0)
        metrics_ctx["completion_tokens"] = int(
            state_out.get("completion_tokens") or 0
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