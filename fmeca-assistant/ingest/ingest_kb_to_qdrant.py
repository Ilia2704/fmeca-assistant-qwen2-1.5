# python3 fmeca-assistant/ingest/ingest_kb_to_qdrant.py

import os
import glob
import uuid
from typing import List, Dict

from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
)


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "fmeca_kb_en")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "..", "kb")


def load_documents(kb_dir: str) -> List[Dict]:
    files = []
    files.extend(glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True))
    files.extend(glob.glob(os.path.join(kb_dir, "**/*.md"), recursive=True))

    documents = []
    for path in sorted(files):
        with open(path, "r", encoding="utf-8") as f:
            documents.append({
                "path": os.path.relpath(path, start=os.path.join(BASE_DIR, "..")),
                "text": f.read().strip()
            })
    return documents


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + CHUNK_SIZE, length)

        if end < length:
            nl = text.rfind("\n", start, end)
            if nl > start + CHUNK_SIZE * 0.6:
                end = nl

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(end - CHUNK_OVERLAP, end)

    return chunks


def build_chunks(documents: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in documents:
        parts = chunk_text(doc["text"])
        for idx, part in enumerate(parts):
            all_chunks.append({
                "text": part,
                "source_file": doc["path"],
                "chunk_index": idx,
            })
    return all_chunks



def main():
    print("Loading documents...")
    documents = load_documents(KB_DIR)
    if not documents:
        raise RuntimeError("No documents found in kb/")

    print("Chunking documents...")
    chunks = build_chunks(documents)

    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    vector_size = model.get_sentence_embedding_dimension()

    # Create collection if needed
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        print(f"Creating collection '{COLLECTION_NAME}'")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    print("Generating embeddings...")
    texts = [f"passage: {c['text']}" for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Uploading to Qdrant...")
    points = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "language": "en",
                    "domain": "fmeca",
                    "node_type": "knowledge_chunk",
                },
            )
        )

    BATCH_SIZE = 128
    for i in tqdm(range(0, len(points), BATCH_SIZE)):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + BATCH_SIZE],
        )

    info = client.get_collection(COLLECTION_NAME)
    print("Done")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Points: {info.points_count}")


if __name__ == "__main__":
    main()