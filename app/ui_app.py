# app/ui_app.py
import streamlit as st
# from retrievers import DenseRetriever
from hybrid_retriever import HybridRetriever  # use hybrid if you added it

st.set_page_config(page_title='DocuQA – FastAPI', layout='wide')
st.title('DocuQA – FastAPI Documentation Q&A')

if 'retr' not in st.session_state:
    st.session_state.retr = HybridRetriever(topk=5, use_reranker=True, rerank_topn=20)

q = st.chat_input('Ask about FastAPI...')
if q:
    with st.spinner('Retrieving...'):
        results = st.session_state.retr.search(q)
    # build a simple context blob
    ctx = []
    for i, (doc, meta) in enumerate(results, 1):
        title = meta.get("title") or meta.get("source", "unknown")
        url = meta.get("url")
        header = f"[Source {i}] {title}"
        with st.expander(header):
            preview = (doc or "").strip()
            if len(preview) > 800:
                preview = preview[:800] + " …"
            st.write(preview)
            if url:
                st.markdown(f"[Open page]({url})", help="Open the source in your browser")
        ctx.append(f"[{i}] {doc}")

    st.info("💡 LLM call goes here (OpenAI/Ollama). Use ctx to answer and cite [1], [2], …")