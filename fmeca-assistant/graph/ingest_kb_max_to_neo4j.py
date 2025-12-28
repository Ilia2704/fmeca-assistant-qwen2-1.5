import os, json, re, hashlib, argparse
from dotenv import load_dotenv
from neo4j import GraphDatabase

def rel(k):  # key -> rel type
    k = re.sub(r"[^A-Za-z0-9]+", "_", str(k)).strip("_").upper() or "FIELD"
    if not re.match(r"^[A-Z]", k): k = "F_" + k
    return "HAS_" + k

def uid(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()

def merge_node(tx, label, props):
    tx.run(f"MERGE (n:{label} {{uid:$uid}}) SET n += $p", uid=props["uid"], p=props)

def merge_rel(tx, a, b, r, p=None):
    tx.run(f"""
        MATCH (x {{uid:$a}}) MATCH (y {{uid:$b}})
        MERGE (x)-[t:{r}]->(y)
        SET t += $p
    """, a=a, b=b, p=p or {})

def walk(tx, parent_uid, key, val, path):
    r = rel(key)

    if isinstance(val, dict):
        cid = uid("obj", path, key)
        merge_node(tx, "JObj", {"uid": cid, "path": path, "key": str(key)})
        merge_rel(tx, parent_uid, cid, r)
        for k, v in val.items():
            walk(tx, cid, k, v, f"{path}/{k}")
        return

    if isinstance(val, list):
        aid = uid("arr", path, key)
        merge_node(tx, "JArr", {"uid": aid, "path": path, "key": str(key), "len": len(val)})
        merge_rel(tx, parent_uid, aid, r)
        for i, it in enumerate(val):
            ikey, ipath = "item", f"{path}[{i}]"
            if isinstance(it, dict):
                iid = uid("obj", ipath, ikey)
                merge_node(tx, "JObj", {"uid": iid, "path": ipath, "key": ikey})
                merge_rel(tx, aid, iid, "ITEM", {"index": i})
                for k, v in it.items():
                    walk(tx, iid, k, v, f"{ipath}/{k}")
            elif isinstance(it, list):
                iid = uid("arr", ipath, ikey)
                merge_node(tx, "JArr", {"uid": iid, "path": ipath, "key": ikey, "len": len(it)})
                merge_rel(tx, aid, iid, "ITEM", {"index": i})
                for j, sub in enumerate(it):
                    walk(tx, iid, "item", sub, f"{ipath}[{j}]")
            else:
                vid = uid("val", ipath, ikey, json.dumps(it, ensure_ascii=False))
                merge_node(tx, "JVal", {"uid": vid, "path": ipath, "key": ikey, "v": it, "t": type(it).__name__})
                merge_rel(tx, aid, vid, "ITEM", {"index": i})
        return

    vid = uid("val", path, key, json.dumps(val, ensure_ascii=False))
    merge_node(tx, "JVal", {"uid": vid, "path": path, "key": str(key), "v": val, "t": type(val).__name__})
    merge_rel(tx, parent_uid, vid, r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--root", default="kb")
    args = ap.parse_args()

    load_dotenv()
    uri, user, pwd = os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, pwd]): raise RuntimeError("Missing NEO4J_* in env")

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    root_uid = uid("root", "/", args.root)
    def txfn(tx):
        tx.run("MERGE (r:JRoot {uid:$uid}) SET r.key=$k, r.path='/'", uid=root_uid, k=args.root)
        walk(tx, root_uid, args.root, data, f"/{args.root}")

    with driver.session() as s:
        s.execute_write(txfn)

    driver.close()
    print("Done: JSON ingested as JRoot/JObj/JArr/JVal graph.")

if __name__ == "__main__":
    main()
