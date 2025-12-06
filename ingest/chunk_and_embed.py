# ingest/chunk_and_embed.py
import os, glob, pathlib, json
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from utils_text import header_aware_chunks

# Resolve paths relative to repo root (so it works no matter where you run it)
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Use your winning model by default
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/e5-small-v2")
COLLECTION = os.getenv("COLLECTION_NAME", "fastapi-docs")

def _doc_prefix_for(model_id: str) -> str:
    mid = (model_id or "").lower()
    if mid.startswith("intfloat/e5") or mid.startswith("baai/bge-"):
        return "passage: "
    return ""

if __name__ == "__main__":
    print(f"[ingest] RAW_DIR={RAW_DIR} | CHROMA_DIR={CHROMA_DIR}")
    files = sorted(glob.glob(str(RAW_DIR / "*.md")))
    print(f"[ingest] raw files found: {len(files)}")
    if not files:
        raise SystemExit("[ingest] No raw files found. Run the crawler first.")

    model = SentenceTransformer(EMB_MODEL)
    DOC_PREFIX = _doc_prefix_for(EMB_MODEL)
    print(f"[embed] EMB_MODEL={EMB_MODEL} | DOC_PREFIX='{DOC_PREFIX}'")

    client = PersistentClient(path=str(CHROMA_DIR))
    try:
        coll = client.get_collection(COLLECTION)
    except Exception:
        coll = client.create_collection(COLLECTION)

    total_chunks = 0
    # load manifest for richer metadata (title, url)
    man_path = RAW_DIR / "_manifest.json"
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    except Exception:
        manifest = {}

    for fp in files:
        text = pathlib.Path(fp).read_text(encoding="utf-8")
        chunks = [c.strip() for c in header_aware_chunks(text) if c.strip()]
        if not chunks:
            print(f"[skip] no chunks -> {fp}")
            continue

        ids = [f"{pathlib.Path(fp).stem}-{i}" for i, _ in enumerate(chunks)]
        embs = model.encode([DOC_PREFIX + c for c in chunks], normalize_embeddings=True).tolist()
        meta = manifest.get(pathlib.Path(fp).name, {})
        title = meta.get("title")
        url = meta.get("url")
        coll.upsert(
            ids=ids,
            embeddings=embs,
            documents=chunks,
            metadatas=[{"source": str(fp), "title": title, "url": url}] * len(chunks)
        )
        total_chunks += len(chunks)
        print(f"[upserted] {pathlib.Path(fp).name} :: {len(chunks)} chunks")

    # final count
    try:
        count = coll.count()
    except Exception:
        count = "?"
    print(f"[done] total chunks upserted: {total_chunks} | collection={COLLECTION} | count={count}")