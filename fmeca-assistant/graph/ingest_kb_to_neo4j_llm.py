#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import time
from typing import List, Tuple, Optional, Dict, Any

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

LLM_MODEL   = os.getenv("LLM_MODEL", "Qwen/Qwen2-1.5B-Instruct")
KB_ROOT     = os.getenv("KB_ROOT", "fmeca-assistant/kb")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "300"))
DEBUG_RAW = os.getenv("DEBUG_RAW", "1").strip().lower() in ("1", "true", "yes")


def log(msg: str):
    print(msg, flush=True)


def load_kb_files(root: str) -> List[Tuple[str, str]]:
    files = []
    for ext in ("*.txt", "*.md"):
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    out = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8") as f:
            out.append((fp, f.read()))
    return out

def chunk_text(text: str, max_chars: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


SYSTEM_PROMPT = """You are an entity extractor for an FMECA knowledge base.

OUTPUT FORMAT (MANDATORY):
Return ONLY one JSON object wrapped in tags exactly:
<json>{"entities":[...]} </json>
No other text.

WHAT COUNTS AS AN ENTITY:
- FMECA/FMEA terminology: Function, Failure Mode, Cause, Effect, Control, Action, RPN, Severity, Occurrence, Detection, S/O/D.
- Items from lists and steps (e.g. "Define scope and system boundaries", "Risk assessment").
- Table headers and column names (for markdown tables: the text between pipes | ... |).
- Glossary terms (patterns like "Term:" or "Term -") and key phrases from their definitions.
- Scale labels and categories (e.g., "Hazardous", "System shutdown", "Performance degradation", "Almost inevitable", "No detection possible").

EXTRACTION RULES:
1) NEVER return an empty list if the text is non-empty.
   If unsure, output at least 10 entities by extracting nouns/phrases directly from the text.
2) Prefer copying phrases verbatim from the text.
3) Entities must be short: 1–6 words. No trailing punctuation. No duplicates.
4) For glossary lines "X: Y", ALWAYS include X as an entity.
5) For numbered steps "1. ...", include the core step phrase as an entity.
6) For markdown table header row, include each header cell as an entity.

QUALITY:
- Return 10–40 entities depending on text size.
- Do not explain. Do not output markdown. Only <json>...</json>.

"""

_JSON_TAG_RE = re.compile(r"<json>\s*(\{.*?\})\s*</json>", re.S)

def parse_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    m = _JSON_TAG_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    i = text.find("{")
    if i == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def init_llm():
    log("Init LLM...")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    return tok, model

def extract_entities(tok, model, text: str) -> List[str]:
    user_prompt = f"Text:\n{text}\n\nReturn ONLY <json>{{...}}</json>."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get(
        "attention_mask", torch.ones_like(input_ids)
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    dt = time.time() - t0

    gen_ids = out[0][input_ids.shape[-1]:]
    decoded = tok.decode(gen_ids, skip_special_tokens=True).strip()

    data = parse_json(decoded)
    if not data:
        log(f"JSON parse failed (gen {dt:.2f}s)")
        if DEBUG_RAW:
            log("RAW OUTPUT:")
            log(decoded)
            log("END RAW OUTPUT")
        return []

    ents = data.get("entities")
    if not isinstance(ents, list):
        log(f"Parsed JSON but 'entities' is not a list (gen {dt:.2f}s)")
        if DEBUG_RAW:
            log("PARSED JSON:")
            log(json.dumps(data, ensure_ascii=False))
            log("END PARSED JSON")
        return []

    cleaned, seen = [], set()
    for e in ents:
        if not isinstance(e, str):
            continue
        s = e.strip().strip(" \t\r\n,.;:()[]{}\"'")
        if not s:
            continue
        if len(s.split()) > 6:
            s = " ".join(s.split()[:6])
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)

    log(f"Parsed {len(cleaned)} entities (gen {dt:.2f}s)")
    return cleaned


def main():
    tok, model = init_llm()

    files = load_kb_files(KB_ROOT)
    log(f"KB files: {len(files)}")

    total = 0
    for fidx, (fp, text) in enumerate(files, start=1):
        rel = os.path.relpath(fp, KB_ROOT)
        chunks = chunk_text(text, CHUNK_CHARS)

        log("")
        log(f"[{fidx}/{len(files)}] File: {rel}")
        log(f"Chunks: {len(chunks)} | Chars: {len(text)}")

        for i, ch in enumerate(chunks, start=1):
            log(f"  Chunk {i}/{len(chunks)} | {len(ch)} chars")
            ents = extract_entities(tok, model, ch)
            total += len(ents)

            if ents:
                log("  Entities:")
                for e in ents[:30]:
                    log(f"    - {e}")
            else:
                log("  Entities: NONE")

    log("")
    log(f"Done. Total extracted entities: {total}")


if __name__ == "__main__":
    main()
