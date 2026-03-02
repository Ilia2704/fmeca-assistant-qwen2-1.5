import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
#from db_clients import get_neo4j_driver

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fmeca_kb_en")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

_client: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(QDRANT_URL)
    return _client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def qdrant_search(query: str, *, top_k: int = 5) -> List[Dict[str, Any]]:
    client = get_qdrant()
    collection = QDRANT_COLLECTION
    emb = get_embedder()

    qvec = emb.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()

    res = client.query_points(
        collection_name=collection,
        query=qvec,              
        limit=top_k,
        with_payload=True,
        with_vectors=False      
    )
    hits = res.points         

    out: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        text = (
            payload.get("text")
            or payload.get("chunk")
            or payload.get("content")
            or ""
        )
        source = (
            payload.get("source")
            or payload.get("file")
            or payload.get("path")
            or "qdrant"
        )
        out.append(
            {
                "score": float(h.score),
                "text": text,
                "source": source,
            }
        )
    return out

def neo4j_hint(query: str, *, limit: int = 10) -> List[str]:
    """
    Neo4j hints are disabled in this demo build.
    We keep the function for backward compatibility, but it does not query Neo4j.
    """
    return []
'''
def neo4j_hint(query: str, *, limit: int = 10) -> List[str]:
    """
    Минимальная подсказка из графа Neo4j:
    - ищем любые узлы, у которых есть name или title,
    - фильтруем по подстроке (case-insensitive),
    - возвращаем список строк вида: ['[Label1,Label2]:имя', ...]
    """
    driver = get_neo4j_driver()

    cypher = """
    MATCH (n)
    WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($q))
    OR (n.title IS NOT NULL AND toLower(n.title) CONTAINS toLower($q))
    RETURN labels(n) AS labels, coalesce(n.name, n.title) AS name
    LIMIT $limit
    """

    rows: List[str] = []
    with driver.session() as s:
        for r in s.run(cypher, q=query, limit=limit):
            labels = r["labels"] or []
            name = r["name"]
            if name:
                rows.append(f"{labels}:{name}")
    return rows

'''
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