# app/hybrid_retriever.py
import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/e5-small-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
COLLECTION = os.getenv("COLLECTION_NAME", "fastapi-docs")

def _prep_query(q: str) -> str:
    m = EMB_MODEL.lower()
    if m.startswith("intfloat/e5") or m.startswith("baai/bge-"):
        return "query: " + q
    return q

class HybridRetriever:
    def __init__(self, topk=8, page_size=1000, use_reranker=False, rerank_topn=20, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.client = PersistentClient(path=CHROMA_DIR)
        self.coll = self.client.get_collection(COLLECTION)
        self.model = SentenceTransformer(EMB_MODEL)
        self.topk = topk
        self.use_reranker = use_reranker
        self.rerank_topn = rerank_topn
        self.reranker = CrossEncoder(reranker_model) if use_reranker else None

        # build BM25 corpus by paging through the collection
        total = self.coll.count()
        docs, metas = [], []
        for offset in range(0, total, page_size):
            batch = self.coll.get(include=["documents","metadatas"], limit=page_size, offset=offset)
            docs += batch["documents"]
            metas += batch["metadatas"]
        self.docs = docs
        self.metas = metas
        self.bm25 = BM25Okapi([d.split() for d in docs])

    def _dedup_by_source(self, pairs, k=None):
        seen, out = set(), []
        limit = k or self.topk
        for doc, meta in pairs:
            src = (meta or {}).get("source")
            if src in seen:
                continue
            out.append((doc, meta))
            seen.add(src)
            if len(out) >= limit:
                break
        return out

    def search(self, query: str):
        # dense
        qv = self.model.encode([_prep_query(query)], normalize_embeddings=True).tolist()
        dres = self.coll.query(query_embeddings=qv, n_results=self.topk*3, include=["documents","metadatas"])
        dense = list(zip(dres["documents"][0], dres["metadatas"][0]))

        # sparse
        scores = self.bm25.get_scores(query.split())
        idx = list(reversed(scores.argsort()))[:self.topk*3]
        sparse = [(self.docs[i], self.metas[i]) for i in idx]

        # reciprocal-rank fusion
        def rrf(lst):
            return {id(doc): 1/(60 + i+1) for i, (doc, _) in enumerate(lst)}
        fused = {}
        for lst in (dense, sparse):
            for i, (doc, meta) in enumerate(lst):
                fused[id(doc)] = fused.get(id(doc), 0) + 1/(60 + i+1)

        # map back to original tuples, keep topk unique
        scored = sorted(((s, d, m) for k, s in fused.items() for d, m in dense+sparse if id(d)==k), reverse=True)
        merged = []
        seen_doc = set()
        for _, d, m in scored:
            if id(d) in seen_doc:
                continue
            merged.append((d, m))
            seen_doc.add(id(d))

        # optional cross-encoder re-ranking for sharper ordering
        if self.use_reranker and self.reranker is not None:
            candidates = merged[: self.rerank_topn]
            pairs = [(query, d) for d, _ in candidates]
            scores = self.reranker.predict(pairs)
            merged = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

        # deduplicate by source to keep only best chunk per page
        deduped = self._dedup_by_source(merged, k=self.topk)
        return deduped