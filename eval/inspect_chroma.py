# scripts/inspect_chroma.py
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os

CHROMA_DIR = "data/chroma"
COLLECTION = "fastapi-docs"
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/e5-small-v2")

def _prep_query(q: str) -> str:
    m = EMB_MODEL.lower()
    if m.startswith("intfloat/e5") or m.startswith("baai/bge-"):
        return "query: " + q
    return q

client = PersistentClient(path=CHROMA_DIR)
print("[collections]", [c.name for c in client.list_collections()])

coll = client.get_collection(COLLECTION)
print("[count]", coll.count())

# peek a few docs
peek = coll.peek(3)
for i in range(len(peek["ids"])):
    print(f"\n--- doc {i+1} ---")
    print("id:", peek["ids"][i])
    print("source:", peek["metadatas"][i].get("source"))
    print("snippet:", (peek["documents"][i] or "")[:200])

# simple retrieval test
model = SentenceTransformer(EMB_MODEL)
q = "How do I return custom response headers in FastAPI?"
qv = model.encode([_prep_query(q)], normalize_embeddings=True).tolist()
res = coll.query(query_embeddings=qv, n_results=5)
print("\n=== search results ===")
for i, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0]), 1):
    print(f"[{i}] source:", meta.get("source"))
    print("    snippet:", (doc or "")[:160])
