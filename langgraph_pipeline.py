from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
import logging
import torch
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from retrieval import qdrant_search, neo4j_hint
from tools.translation_yandex import translate as translate_ygpt

log = logging.getLogger("pipeline")

SYSTEM_PROMPT = """
You are a FMECA (Failure Modes, Effects and Criticality Analysis) and reliability engineering assistant
that uses an external knowledge base (KB).

- Always answer in English, clearly and structurally.
- You receive a KB `context`. Treat it as the main source of truth.
- When the context is non-empty and related to the user topic (e.g. pumps, batteries, conveyors),
  use it directly instead of saying there is not enough information.
- Short or vague queries (e.g. a single word like "pump") should be interpreted as a request
  for an FMECA-style description based on the KB context, if it is relevant.

For FMECA-style questions ("failure modes for X", "function of X", "FMECA for X"):
- If the context has relevant fragments for X, extract and present, as far as present in the KB:
  - Function(s)
  - Failure Modes
  - Causes
  - Local / System / End Effects
  - Existing controls
  - S, O, D, RPN
  - Recommended actions
- You may merge or lightly rephrase KB entries, but DO NOT invent new failure modes, causes,
  effects or numerical ratings that are not supported by the context.

Insufficient or empty context:
- Only say that the context is insufficient if it is empty or clearly unrelated to the question.
- If the context is partially relevant, still give your best FMECA-style answer from it and,
  if needed, mention what extra information would refine the analysis.
- If there is no relevant KB at all, you may give generic guidance on how to perform an FMECA
  for such an item, clearly stating that this is general engineering advice, not KB-based.

Never hallucinate detailed FMECA tables or specific numeric values that are not supported
by the provided context.
"""


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

    # Tracing / logging
    request_id: str
    logs: List[str]
    node_path: List[str]


def _log(state: PipelineState, message: str) -> None:
    """Append message to in-memory logs and emit to logger with request id.

    Also maintains a simple ordered list of visited nodes in state["node_path"],
    assuming messages are prefixed with "[node_name]".
    """
    logs = list(state.get("logs") or [])
    logs.append(message)
    state["logs"] = logs

    # Update node path from prefix like "[node_name] ..."
    if message.startswith("[") and "]" in message:
        node_name = message[1 : message.index("]")]
        path = list(state.get("node_path") or [])
        if not path or path[-1] != node_name:
            path.append(node_name)
            state["node_path"] = path

    # Emit to structured logger with request id
    rid = state.get("request_id", "no_request_id")
    log.info("[%s] %s", rid, message)


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
    #g_hits = neo4j_hint(query_en, limit=5)
    g_hits: List[str] = []  # neo4j disabled

    parts: List[str] = []
    if q_hits:
        parts.append("KB (semantic matches):")
        for i, h in enumerate(q_hits, 1):
            text = (h.get("text") or "").replace("\n", " ").strip()
            if len(text) > 800:
                text = text[:797] + "..."
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
            "First, if the context already contains an explicit FMECA-style example for the same equipment, "
            "summarize that example (functions, failure modes, causes, effects, controls, S/O/D/RPN, actions). "
            "Only after that, you may optionally add short generic remarks. "
            "Do not invent specific failure modes, causes, effects, or numerical values that are not supported by this context."
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

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

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
    request_id: Optional[str] = None,
    do_sample: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> PipelineState:
    """Run the LangGraph pipeline for a single user query.

    The optional request_id is used only for logging / monitoring.
    If not provided, it is derived from the first 10 characters of user_query.
    """
    base = (user_query or "").strip()
    rid = (request_id or (base[:10] if base else "empty_query"))

    init_state: PipelineState = {
        "user_query": user_query,
        "tokenizer": tokenizer,
        "model": model,
        "history": list(history or []),
        "request_id": rid,
        "logs": [],
        "node_path": [],
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    state_out: PipelineState = PIPELINE_APP.invoke(init_state)

    path = state_out.get("node_path") or []
    log.info(
        "[%s] PIPELINE DONE, path=%s",
        rid,
        " -> ".join(path) if path else "<none>",
    )
    return state_out


def print_graph_ascii() -> None:
    PIPELINE_APP.get_graph().print_ascii()