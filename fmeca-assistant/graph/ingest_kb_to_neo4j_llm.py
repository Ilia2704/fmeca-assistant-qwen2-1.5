import os, json, glob
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

# --- env ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "1234!")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEDUP_COLLECTION = os.getenv("DEDUP_COLLECTION", "fmeca_entities_dedup")

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2-1.5B-Instruct")

KB_ROOT = os.getenv("KB_ROOT", "fmeca-assistant/kb")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.90"))  # cosine

ALLOWED_NODE_TYPES = [
    "Component","Function","FailureMode","Cause","Effect","Control","Action","Concept"
]
ALLOWED_EDGE_TYPES = [
    "MENTIONS","FAILS_AS","CAUSED_BY","LEADS_TO","MITIGATED_BY","IMPROVED_BY","RELATED_TO"
]

SYSTEM_PROMPT = """You extract a knowledge graph from text.
Return ONLY valid JSON. No markdown.

Rules:
- Extract only what is explicitly stated in the text. Do NOT invent.
- Use only allowed node types and edge types.
- Prefer specific FMECA types (Function/FailureMode/Cause/Effect/Control/Action) when text supports it.
- Otherwise use Concept.
- Every edge must include a short evidence span copied from the text (<= 25 words).
- Confidence: 0.0 to 1.0.
JSON schema:
{
  "nodes":[{"type":"...", "name":"..."}],
  "edges":[{"from":"<name>", "from_type":"...", "rel":"...", "to":"<name>", "to_type":"...", "evidence":"...", "confidence":0.0}]
}
"""

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

def load_kb_files(root: str) -> List[Tuple[str, str]]:
    exts = ("*.txt","*.md")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    out = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8") as f:
            out.append((fp, f.read()))
    return out

def init_llm():
    tok = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tok, model

def llm_extract(tok, model, text: str) -> Dict[str, Any]:
    user_prompt = f"Text:\n{text}\n\nAllowed node types: {ALLOWED_NODE_TYPES}\nAllowed edge types: {ALLOWED_EDGE_TYPES}\nReturn JSON only."
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user_prompt}
    ]
    input_ids = tok.apply_chat_template(messages, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=700,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)

    # Extract last JSON object from the decoded text
    start = decoded.rfind("{")
    end = decoded.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"nodes":[], "edges":[]}
    js = decoded[start:end+1]
    try:
        data = json.loads(js)
    except Exception:
        return {"nodes":[], "edges":[]}

    # hard validation
    nodes = []
    for n in data.get("nodes", []):
        t, name = n.get("type"), (n.get("name") or "").strip()
        if t in ALLOWED_NODE_TYPES and name:
            nodes.append({"type":t, "name":name})

    edges = []
    for e in data.get("edges", []):
        rel = e.get("rel")
        if rel not in ALLOWED_EDGE_TYPES:
            continue
        frm, to = (e.get("from") or "").strip(), (e.get("to") or "").strip()
        ft, tt = e.get("from_type"), e.get("to_type")
        ev = (e.get("evidence") or "").strip()
        conf = float(e.get("confidence", 0.5))
        if not frm or not to or ft not in ALLOWED_NODE_TYPES or tt not in ALLOWED_NODE_TYPES:
            continue
        if not ev or len(ev.split()) > 25:
            continue
        edges.append({
            "from":frm, "from_type":ft,
            "to":to, "to_type":tt,
            "rel":rel, "evidence":ev, "confidence":max(0.0, min(1.0, conf))
        })
    return {"nodes":nodes, "edges":edges}

def ensure_qdrant_dedup(client: QdrantClient, dim: int):
    try:
        client.get_collection(DEDUP_COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=DEDUP_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def canonicalize_entity(qclient: QdrantClient, embedder: SentenceTransformer, etype: str, name: str) -> str:
    # vectorize entity name
    vec = embedder.encode([f"query: {name}"], normalize_embeddings=True)[0].tolist()

    # query existing
    try:
        resp = qclient.query_points(
            collection_name=DEDUP_COLLECTION,
            query=vec,
            limit=1,
            with_payload=True
        )
        if resp.points:
            best = resp.points[0]
            score = best.score
            payload = best.payload or {}
            existing = payload.get("canonical_name")
            existing_type = payload.get("type")
            if existing and existing_type == etype and score >= DEDUP_THRESHOLD:
                return existing
    except Exception:
        pass

    # insert new
    pid = f"{etype}:{name}".lower()
    qclient.upsert(
        collection_name=DEDUP_COLLECTION,
        points=[PointStruct(
            id=pid,
            vector=vec,
            payload={"type":etype, "canonical_name":name}
        )]
    )
    return name

def neo4j_upsert(driver, doc: str, chunk_id: str, nodes: List[Dict[str,str]], edges: List[Dict[str,Any]]):
    # We store all entities as (:Entity {type, name}) + label by type for nicer UI
    # And connect Document -> Entity via MENTIONS with provenance.
    cypher = """
    MERGE (d:Document {path:$doc})
    SET d.updated_at=timestamp()

    WITH d
    UNWIND $nodes AS n
      MERGE (e:Entity {type:n.type, name:n.name})
      SET e:`${DUMMY}` = true
    """
    # Can't parameterize labels in Cypher directly; we'll create typed labels in a second query.

    with driver.session() as s:
        s.run("MERGE (d:Document {path:$doc}) SET d.updated_at=timestamp()", doc=doc)

        # create entities
        s.run("""
        UNWIND $nodes AS n
          MERGE (e:Entity {type:n.type, name:n.name})
          SET e.updated_at=timestamp()
        """, nodes=nodes)

        # add typed labels (one-by-one, small KB => ok)
        for n in nodes:
            s.run(f"""
            MATCH (e:Entity {{type:$t, name:$name}})
            SET e:{n["type"]}
            """, t=n["type"], name=n["name"])

        # mentions edges
        s.run("""
        MATCH (d:Document {path:$doc})
        UNWIND $nodes AS n
          MATCH (e:Entity {type:n.type, name:n.name})
          MERGE (d)-[r:MENTIONS]->(e)
          SET r.chunk_id=$chunk_id, r.updated_at=timestamp()
        """, doc=doc, nodes=nodes, chunk_id=chunk_id)

        # semantic edges
        s.run("""
        UNWIND $edges AS e
          MATCH (a:Entity {type:e.from_type, name:e.from})
          MATCH (b:Entity {type:e.to_type, name:e.to})
          MERGE (a)-[r:REL {type:e.rel}]->(b)
          SET r.evidence=e.evidence, r.confidence=e.confidence,
              r.doc=$doc, r.chunk_id=$chunk_id, r.updated_at=timestamp()
        """, edges=edges, doc=doc, chunk_id=chunk_id)

        # add relationship-type labels for nicer browsing (optional)
        # You can later query by r.type.

def main():
    print("Loading embedder...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Init Qdrant dedup collection...")
    qclient = QdrantClient(url=QDRANT_URL)
    ensure_qdrant_dedup(qclient, dim=embedder.get_sentence_embedding_dimension())

    print("Init LLM...")
    tok, model = init_llm()

    print("Connect Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    files = load_kb_files(KB_ROOT)
    print(f"KB files: {len(files)}")

    total_edges = 0
    for fp, text in files:
        chunks = chunk_text(text, CHUNK_CHARS)
        for i, ch in enumerate(chunks):
            chunk_id = f"{os.path.relpath(fp, KB_ROOT)}::{i}"
            data = llm_extract(tok, model, ch)

            # canonicalize nodes
            nodes = []
            seen = set()
            for n in data["nodes"]:
                canon = canonicalize_entity(qclient, embedder, n["type"], n["name"])
                key = (n["type"], canon)
                if key in seen:
                    continue
                seen.add(key)
                nodes.append({"type": n["type"], "name": canon})

            # canonicalize edges (names too)
            edges = []
            for e in data["edges"]:
                frm = canonicalize_entity(qclient, embedder, e["from_type"], e["from"])
                to = canonicalize_entity(qclient, embedder, e["to_type"], e["to"])
                edges.append({**e, "from": frm, "to": to})

            if nodes or edges:
                neo4j_upsert(driver, doc=os.path.relpath(fp, KB_ROOT), chunk_id=chunk_id, nodes=nodes, edges=edges)
                total_edges += len(edges)

    driver.close()
    print(f"Done. Total semantic edges: {total_edges}")

if __name__ == "__main__":
    main()
