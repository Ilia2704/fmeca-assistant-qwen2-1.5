from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import torch
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from retrieval import qdrant_search, neo4j_hint
from tools.translation_yandex import translate as translate_ygpt

SYSTEM_PROMPT = (
    "You are a FMECA / reliability engineering assistant. "
    "Use the provided knowledge base context when available. "
    "If the context is insufficient, say that you are not sure "
    "and explicitly state what information is missing."
)


def wrap_braces(text: str) -> str:
    """Helper used to wrap strings in braces for logging/formatting."""
    return "{" + text + "}"

class PipelineState(TypedDict, total=False):
    user_query: str
    detected_lang: Literal["ru", "en"]
    query_en: str

    retrieved_chunks: List[Dict[str, Any]]
    graph_hints: List[str]
    context: str

    answer_en: str
    answer_ru: str

    do_sample: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int

    tokenizer: Any
    model: Any
    history: List[Dict[str, str]]

    logs: List[str]


def _log(state: PipelineState, message: str) -> None:
    logs = list(state.get("logs") or [])
    logs.append(message)
    state["logs"] = logs


def detect_language(state: PipelineState) -> PipelineState:
    text = (state.get("user_query") or "").strip()
    has_cyr = any("\u0400" <= ch <= "\u04FF" for ch in text)
    lang: Literal["ru", "en"] = "ru" if has_cyr else "en"
    state["detected_lang"] = lang
    if lang == "en":
        state["query_en"] = text
    _log(state, f"[detect_language] detected_lang={lang}")
    return state


def translate_ru_en(state: PipelineState) -> PipelineState:
    text = (state.get("user_query") or "").strip()
    query_en = translate_ygpt(text, target_lang="en", source_lang="ru")
    state["query_en"] = query_en
    _log(state, "[translate_ru_en] RU -> EN done")
    return state


def retrieval_node(state: PipelineState) -> PipelineState:
    query_en = (state.get("query_en") or state.get("user_query") or "").strip()
    if not query_en:
        state["retrieved_chunks"] = []
        state["graph_hints"] = []
        state["context"] = ""
        _log(state, "[retrieval] empty query")
        return state

    q_hits = qdrant_search(query_en, top_k=5)
    g_hits = neo4j_hint(query_en, limit=5)

    parts: List[str] = []
    if q_hits:
        parts.append("KB (semantic matches):")
        for i, h in enumerate(q_hits, 1):
            text = (h.get("text") or "").replace("\n", " ").strip()
            if len(text) > 300:
                text = text[:297] + "..."
            src = h.get("source")
            score = h.get("score", 0.0)
            parts.append(f"{i}. {text} (src={src}, score={score:.3f})")

    if g_hits:
        parts.append("Graph hints (name/title matches):")
        for i, x in enumerate(g_hits, 1):
            parts.append(f"{i}. {x}")

    ctx = "\n".join(parts).strip()

    state["retrieved_chunks"] = q_hits
    state["graph_hints"] = g_hits
    state["context"] = ctx
    _log(state, f"[retrieval] q_hits={len(q_hits)}, g_hits={len(g_hits)}")
    return state


def generate_en(state: PipelineState) -> PipelineState:
    tokenizer = state.get("tokenizer")
    model = state.get("model")
    history: List[Dict[str, str]] = list(state.get("history") or [])
    if tokenizer is None or model is None:
        raise RuntimeError("tokenizer/model not in state")

    query_en = (state.get("query_en") or state.get("user_query") or "").strip()
    context = (state.get("context") or "").strip()

    do_sample = bool(state.get("do_sample", False))
    max_new_tokens = int(state.get("max_new_tokens", 256))
    temperature = float(state.get("temperature", 0.7))
    top_p = float(state.get("top_p", 0.9))
    top_k = int(state.get("top_k", 50))

    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(history)

    if context:
        user_content = (
            "You are a FMECA / reliability engineering assistant.\n\n"
            f"User question (EN): {query_en}\n\n"
            "Use the following knowledge base context when answering:\n"
            f"{context}\n\n"
            "If the context is insufficient, say that you are not sure and what information is missing."
        )
    else:
        user_content = (
            "You are a FMECA / reliability engineering assistant.\n\n"
            f"User question (EN): {query_en}\n\n"
            "Answer briefly and clearly in English."
        )

    messages.append({"role": "user", "content": user_content})

    device = model.device
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    prompt_len = input_ids.shape[-1]
    gen_ids = output_ids[0, prompt_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    answer_en = wrap_braces(raw)

    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": answer_en})

    state["answer_en"] = answer_en
    state["history"] = history
    _log(state, f"[generate_en] answer_en_len={len(answer_en)}")
    return state


def translate_en_ru(state: PipelineState) -> PipelineState:
    answer_en = (state.get("answer_en") or "").strip()
    if not answer_en:
        _log(state, "[translate_en_ru] no EN answer")
        return state

    inner = answer_en
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1].strip()

    answer_ru_plain = translate_ygpt(inner, target_lang="ru", source_lang="en")
    answer_ru = wrap_braces(answer_ru_plain)
    state["answer_ru"] = answer_ru
    _log(state, "[translate_en_ru] EN -> RU done")
    return state


def build_pipeline_graph():
    g = StateGraph(PipelineState)
    g.add_node("detect_language", detect_language)
    g.add_node("maybe_translate_ru_en", translate_ru_en)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generate_en", generate_en)
    g.add_node("translate_en_ru", translate_en_ru)

    g.set_entry_point("detect_language")
    g.add_conditional_edges(
        "detect_language",
        lambda s: s["detected_lang"],
        {"ru": "maybe_translate_ru_en", "en": "retrieval"},
    )
    g.add_edge("maybe_translate_ru_en", "retrieval")
    g.add_edge("retrieval", "generate_en")
    g.add_edge("generate_en", "translate_en_ru")
    g.add_edge("translate_en_ru", END)
    return g.compile()


PIPELINE_APP = build_pipeline_graph()


def run_pipeline(
    user_query: str,
    tokenizer: Any,
    model: Any,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    do_sample: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> PipelineState:
    init_state: PipelineState = {
        "user_query": user_query,
        "tokenizer": tokenizer,
        "model": model,
        "history": list(history or []),
        "logs": [],
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    return PIPELINE_APP.invoke(init_state)


def print_graph_ascii() -> None:
    PIPELINE_APP.get_graph().print_ascii()