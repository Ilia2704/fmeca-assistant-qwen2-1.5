# fmeca-assistant/ingest/query_kb.py
#
# Interactive semantic search over Qdrant collection.
# Works with the same embedding model as ingest_kb_to_qdrant.py (E5-style).
#
# Usage:
#   source .venv/bin/activate
#   python3 fmeca-assistant/ingest/query_kb.py
#
# Env (.env):
#   QDRANT_URL=http://localhost:6333
#   QDRANT_COLLECTION=fmeca_kb_en
#   EMBED_MODEL=intfloat/multilingual-e5-small

import os
from typing import Any, Dict

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "fmeca_kb_en")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))


def _short(text: str, n: int = 450) -> str:
    text = (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    text_one_line = " ".join(text.split())
    return text_one_line[:n] + ("..." if len(text_one_line) > n else "")


def _safe_payload(payload: Any) -> Dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Embed model: {EMBED_MODEL}")
    print(f"Top-K: {TOP_K_DEFAULT}")

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBED_MODEL)

    # Quick collection existence check (friendly error)
    try:
        info = client.get_collection(COLLECTION)
        print(f"Collection points: {info.points_count}")
    except Exception as e:
        raise SystemExit(
            f"Collection '{COLLECTION}' not found or Qdrant is unreachable.\n"
            f"Details: {e}\n"
            f"Tip: run ingest_kb_to_qdrant.py first, and ensure Qdrant is running."
        )

    print("\nType a question and press Enter. Type 'exit' to quit.\n")

    while True:
        q = input("Query> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break

        # E5 convention: prefix queries with "query: "
        qvec = model.encode([f"query: {q}"], normalize_embeddings=True)[0].tolist()

        response = client.query_points(
            collection_name=COLLECTION,
            query=qvec,
            limit=TOP_K_DEFAULT,
            with_payload=True,
        )

        results = response.points

        if not results:
            print("No results.\n")
            continue

        print("\nTop results:")
        for i, r in enumerate(results, 1):
            payload = _safe_payload(r.payload)
            src = payload.get("source_file", "?")
            idx = payload.get("chunk_index", "?")
            txt = payload.get("text", "")
            print(f"\n{i}) score={r.score:.4f} | src={src} | chunk={idx}")
            print(_short(txt))

        print("\n")


if __name__ == "__main__":
    main()
