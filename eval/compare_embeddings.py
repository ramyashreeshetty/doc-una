# eval/compare_embeddings.py
import os, time, json, glob, pathlib, chromadb, numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PERSIST_DIR = DATA_DIR / "chroma"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# === choose models to compare (all free) ===
CANDIDATES = [
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
    ("e5-small-v2", "intfloat/e5-small-v2"),
]

GT_QA = pathlib.Path("eval/gt_qa.jsonl")  # optional (built via build_ground_truth.py)

def load_chunks() -> List[Dict]:
    rows = []
    for fp in glob.glob(str(RAW_DIR / "*.md")):
        text = pathlib.Path(fp).read_text(encoding="utf-8")
        # lazy import to avoid dependency here; use simple paragraph split for fairness
        import re
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        for i, c in enumerate(paras):
            rows.append({
                "id": f"{pathlib.Path(fp).stem}-p{i}",
                "text": c,
                "source": pathlib.Path(fp).name
            })
    return rows

def build_collection(client, name: str, model_name: str, corpus: List[Dict]):
    # (re)create collection cleanly for each model
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.get_or_create_collection(name=name)
    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()
    # batch upsert
    B = 256
    ids, docs, metas = [], [], []
    for i, r in enumerate(corpus):
        ids.append(r["id"]); docs.append(r["text"]); metas.append({"source": r["source"]})
        if len(ids) == B or i == len(corpus)-1:
            embs = st_model.encode(docs, normalize_embeddings=True).tolist()
            coll.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
            ids, docs, metas = [], [], []
    return coll, dim

def load_gt() -> List[Tuple[str, str]]:
    pairs = []
    if GT_QA.exists():
        for line in GT_QA.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            # use question & known source filename from your generator
            pairs.append((obj["question"], obj.get("source","")))
    else:
        # weak eval: synthesize questions from filenames
        for fp in glob.glob(str(RAW_DIR / "*.md")):
            name = pathlib.Path(fp).name
            q = f"What does the FastAPI page '{name.replace('.md','')}' cover?"
            pairs.append((q, name))
    return pairs

def recall_at_k(targets: List[str], preds: List[List[str]], k=5) -> float:
    hit = 0
    for tgt, p in zip(targets, preds):
        if any(tgt in m for m in p[:k]):
            hit += 1
    return hit / max(1, len(targets))

def mrr_at_k(targets: List[str], preds: List[List[str]], k=5) -> float:
    s = 0.0
    for tgt, p in zip(targets, preds):
        rr = 0.0
        for i, m in enumerate(p[:k]):
            if tgt in m:
                rr = 1.0 / (i+1)
                break
        s += rr
    return s / max(1, len(targets))

def main():
    corpus = load_chunks()
    print(f"Loaded corpus paragraphs: {len(corpus)}")
    qa = load_gt()
    print(f"Loaded eval pairs: {len(qa)} (question, target_source_file)")

    client = chromadb.Client(Settings(persist_directory=str(PERSIST_DIR)))

    results = []
    for short, model_id in CANDIDATES:
        coll_name = f"fastapi-{short}"
        coll, dim = build_collection(client, coll_name, model_id, corpus)

        q_texts = [q for q,_ in qa]
        targets = [t for _,t in qa]

        preds = []
        t0 = time.time()
        B = 64
        # embed queries once
        st_model = SentenceTransformer(model_id)
        for i in range(0, len(q_texts), B):
            batch = q_texts[i:i+B]
            qv = st_model.encode(batch, normalize_embeddings=True).tolist()
            out = coll.query(query_embeddings=qv, n_results=5)   # dense-only for fair embed comparison
            # map back to sources (filenames) from metadata
            for j in range(len(batch)):
                metas = [m.get("source","") for m in out["metadatas"][j]]
                preds.append(metas)
        t_latency = (time.time() - t0) / max(1, len(q_texts))

        r5 = recall_at_k(targets, preds, k=5)
        m5 = mrr_at_k(targets, preds, k=5)
        results.append((short, dim, r5, m5, t_latency))

    print("\n=== Embedding Comparison (dense-only) ===")
    print(f"{'model':22} {'dim':>4}  {'Recall@5':>9} {'MRR@5':>8} {'avg_ms/query':>12}")
    for short, dim, r5, m5, lat in results:
        print(f"{short:22} {dim:4d}  {r5:9.3f} {m5:8.3f} {lat*1000:12.1f}")

if __name__ == "__main__":
    main()
