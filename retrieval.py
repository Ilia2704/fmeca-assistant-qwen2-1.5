# retrieval.py
import os
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter

from db_clients import get_qdrant, get_neo4j_driver

_EMB_MODEL = None

def get_embedder() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        name = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
        _EMB_MODEL = SentenceTransformer(name)
    return _EMB_MODEL

def qdrant_search(query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
    client = get_qdrant()
    collection = os.getenv("QDRANT_COLLECTION", "fmeca_kb_en")
    emb = get_embedder()

    # E5-style: желательно префиксировать запрос
    qvec = emb.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()

    hits = client.search(
        collection_name=collection,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=None,  # при желании можно фильтровать по source/type
    )

    out = []
    for h in hits:
        payload = h.payload or {}
        text = payload.get("text") or payload.get("chunk") or payload.get("content") or ""
        source = payload.get("source") or payload.get("file") or payload.get("path") or "qdrant"
        out.append({"score": float(h.score), "text": text, "source": source})
    return out

def neo4j_hint(query: str, *, limit: int = 10) -> List[str]:
    """
    Минимальная “подсказка” из графа:
    - если у тебя есть узлы Entity/FailureMode/Component и т.п., можно вытаскивать совпадения по name.
    Подстрой под свою схему.
    """
    driver = get_neo4j_driver()

    cypher = """
    MATCH (n)
    WHERE (exists(n.name) AND toLower(n.name) CONTAINS toLower($q))
       OR (exists(n.title) AND toLower(n.title) CONTAINS toLower($q))
    RETURN labels(n) AS labels, coalesce(n.name, n.title) AS name
    LIMIT $limit
    """
    rows = []
    with driver.session() as s:
        for r in s.run(cypher, q=query, limit=limit):
            labels = r["labels"] or []
            name = r["name"]
            if name:
                rows.append(f"{labels}:{name}")
    return rows

def build_context(user_text: str, *, qdrant_k: int = 5, neo4j_k: int = 10) -> str:
    q_hits = qdrant_search(user_text, top_k=qdrant_k)
    g_hits = neo4j_hint(user_text, limit=neo4j_k)

    parts = []
    if q_hits:
        parts.append("KB (semantic matches):")
        for i, h in enumerate(q_hits, 1):
            t = (h["text"] or "").strip()
            if t:
                t = t.replace("\n", " ").strip()
                parts.append(f"{i}. {t} (src={h['source']}, score={h['score']:.3f})")

    if g_hits:
        parts.append("Graph hints (name/title matches):")
        for i, x in enumerate(g_hits, 1):
            parts.append(f"{i}. {x}")

    return "\n".join(parts).strip()
