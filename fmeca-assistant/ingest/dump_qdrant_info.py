#!/usr/bin/env python
# dump_qdrant_info.py
#
# 1) Check Qdrant availability
# 2) Optionally update collection params (HNSW + optimizer, clear points)
# 3) Dump Qdrant meta information to logs/qdrant_full_info.json:
#    - server info
#    - collections list
#    - per-collection config (including HNSW and payload schema)
#
# Self-contained: does not import db_clients.

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --------------------------------------------------------------------
# Parameter block: tweak these values for your demo
# --------------------------------------------------------------------

# Target collection to modify (and optionally clear)
TARGET_COLLECTION = os.getenv("QDRANT_COLLECTION", "fmeca_kb_en")

# If True, apply HNSW / optimizer config below to TARGET_COLLECTION
UPDATE_CONFIG = True  # set to True when you want to update

# If True, delete all points from TARGET_COLLECTION (keep schema/config)
CLEAR_COLLECTION = True  # set to True to wipe all points

# Demo HNSW / optimizer settings:
# For tiny collections (25–50 points) you can force HNSW usage by:
#   - full_scan_threshold = 10
#   - indexing_threshold = 0
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
HNSW_FULL_SCAN_THRESHOLD = 10

OPTIMIZER_INDEXING_THRESHOLD = 0

# --------------------------------------------------------------------


def get_qdrant() -> QdrantClient:
    """Create a Qdrant client using QDRANT_URL env variable."""
    return QdrantClient(url=QDRANT_URL)


def to_plain(obj):
    """Convert Qdrant / pydantic models to plain Python types for JSON dumping."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_plain(x) for x in obj]

    # pydantic v2
    if hasattr(obj, "model_dump"):
        return to_plain(obj.model_dump())

    # pydantic v1
    if hasattr(obj, "dict"):
        return to_plain(obj.dict())

    # Fallback: string representation
    return repr(obj)


def clear_collection(client: QdrantClient, collection_name: str) -> None:
    """Delete all points from a collection by scrolling all IDs."""
    print(f"Clearing collection points: {collection_name}")
    all_ids = []
    offset = None

    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        if not records:
            break
        all_ids.extend([r.id for r in records])
        if offset is None:
            break

    if not all_ids:
        print(f"No points to delete in collection '{collection_name}'.")
        return

    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=all_ids),
        wait=True,
    )
    print(f"Deleted {len(all_ids)} points from collection '{collection_name}'.")


def main() -> None:
    client = get_qdrant()

    print(f"Qdrant URL: {QDRANT_URL}")

    # Basic health / server info check: if Qdrant is down, this will raise
    server_info = client.info()
    print("Server info:", server_info)

    # List collections to see if TARGET_COLLECTION exists
    cols_resp_initial = client.get_collections()
    collections_initial = [c.name for c in cols_resp_initial.collections]
    print("Collections:", collections_initial)

    if TARGET_COLLECTION in collections_initial:
        print(f"Target collection found: {TARGET_COLLECTION}")

        # Optional: update HNSW / optimizer config for the target collection
        if UPDATE_CONFIG:
            print("Updating collection config (HNSW + optimizer)...")
            client.update_collection(
                collection_name=TARGET_COLLECTION,
                hnsw_config=models.HnswConfigDiff(
                    m=HNSW_M,
                    ef_construct=HNSW_EF_CONSTRUCT,
                    full_scan_threshold=HNSW_FULL_SCAN_THRESHOLD,
                ),
                optimizer_config=models.OptimizersConfigDiff(
                    indexing_threshold=OPTIMIZER_INDEXING_THRESHOLD,
                ),
            )

        # Optional: clear all points from the target collection
        if CLEAR_COLLECTION:
            clear_collection(client, TARGET_COLLECTION)
    else:
        print(
            f"Target collection '{TARGET_COLLECTION}' not found; "
            "skipping config update and clear."
        )

    # ----------------------------------------------------------------
    # Build JSON dump after all modifications
    # ----------------------------------------------------------------
    data = {}

    # Server-level info (from earlier)
    data["info"] = to_plain(server_info)

    # Collections list (after potential updates/clears)
    cols_resp = client.get_collections()
    data["collections_raw"] = to_plain(cols_resp)
    collections = [c.name for c in cols_resp.collections]

    data["collections"] = {}

    for name in collections:
        # Per-collection config (includes HNSW, vector params, payload schema, counts)
        info = client.get_collection(name)
        data["collections"][name] = to_plain(info)

    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "qdrant_full_info.json"

    text = json.dumps(data, indent=2, ensure_ascii=False)
    out_path.write_text(text, encoding="utf-8")

    # Also print to stdout so you can pipe/grep if needed
    print(text)
    print(f"\nWrote Qdrant meta info to: {out_path.resolve()}")


if __name__ == "__main__":
    main()