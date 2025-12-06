# Doc-Una: FastAPI Documentation Knowledge Base

> **Note:** This project was originally intended as my submission for the **LLM Zoomcamp** course but remained incomplete at the time of the deadline. I'm documenting and planning to complete it as a learning project.

A **Retrieval-Augmented Generation (RAG)** system for intelligent Q&A on FastAPI documentation, featuring hybrid search (semantic + keyword) and vector embeddings.

## 🎯 Project Overview

Doc-Una is designed to provide accurate, context-aware answers to questions about FastAPI by combining:
- **Web scraping** of curated FastAPI documentation
- **Hybrid retrieval** using BM25 (keyword) + semantic search
- **Vector embeddings** with ChromaDB
- **Cross-encoder re-ranking** for precision
- **Streamlit UI** for interactive queries

## ✨ Features

### ✅ Completed
- **Automated Documentation Scraping**: Extracts content from 41 curated FastAPI pages
- **Intelligent Text Chunking**: Header-aware chunking with overlap for context preservation
- **Hybrid Search System**: Combines BM25 and semantic search using Reciprocal Rank Fusion (RRF)
- **Vector Database**: ChromaDB with `intfloat/e5-small-v2` embeddings
- **Cross-Encoder Re-ranking**: Optional precision boost with `ms-marco-MiniLM-L-6-v2`
- **Interactive UI**: Streamlit-based interface with expandable result cards
- **Evaluation Framework**: Ground truth dataset generation and embedding comparison tools
- **Docker Support**: Containerized deployment ready

### 🚧 In Progress
- **LLM Integration**: Answer generation with citations (OpenAI/Ollama/Anthropic)
- **Chat History**: Multi-turn conversation support
- **Evaluation Metrics**: Hit rate, MRR, and NDCG benchmarking

## 🏗️ Architecture

```
┌─────────────────┐
│ FastAPI Docs    │
│ (41 pages)      │
└────────┬────────┘
         │ Web Scraping
         ▼
┌─────────────────┐
│ Raw Markdown    │
│ (data/raw/)     │
└────────┬────────┘
         │ Chunking + Embedding
         ▼
┌─────────────────┐
│ ChromaDB        │
│ Vector Store    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Hybrid Retriever        │
│ • BM25 (sparse)         │
│ • Semantic (dense)      │
│ • RRF Fusion            │
│ • Cross-Encoder Rerank  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit UI    │
│ (Q&A Interface) │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd doc-una

# Setup environment and install dependencies
make setup
```

### Usage

#### 1. Ingest Documentation
```bash
make ingest
```
This will:
- Scrape 41 FastAPI documentation pages
- Save raw content to `data/raw/`
- Chunk text and generate embeddings
- Store vectors in ChromaDB (`data/chroma/`)

#### 2. Run the Application
```bash
make app
```
Opens Streamlit UI at `http://localhost:8501`

#### 3. Docker Deployment
```bash
make docker
```

## 📁 Project Structure

```
doc-una/
├── app/
│   ├── hybrid_retriever.py    # Hybrid search engine
│   └── ui_app.py               # Streamlit interface
├── ingest/
│   ├── config.yaml             # URLs to scrape
│   ├── crawl_docs.py           # Web scraper
│   ├── chunk_and_embed.py      # Chunking + embedding
│   └── utils_text.py           # Text processing
├── eval/
│   ├── build_ground_truth.py   # Q&A dataset generator
│   ├── compare_embeddings.py   # Model benchmarking
│   ├── inspect_chroma.py       # DB inspection
│   └── gt_qa.{csv,jsonl}       # Ground truth data
├── data/
│   ├── raw/                    # Scraped markdown (41 files)
│   └── chroma/                 # Vector database
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .docs/
│   └── project_understanding.md # Detailed documentation
├── requirements.txt
├── Makefile
└── README.md
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Web Scraping** | BeautifulSoup, Requests |
| **Embeddings** | Sentence-Transformers (`intfloat/e5-small-v2`) |
| **Vector DB** | ChromaDB |
| **Keyword Search** | BM25 (rank-bm25) |
| **Re-ranking** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **UI** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Testing** | Pytest |

## 🔍 How It Works

### 1. **Data Ingestion**
- Scrapes curated FastAPI documentation pages (tutorials, advanced topics, security)
- Extracts clean content (removes navigation, footers)
- Preserves metadata (titles, URLs) in `_manifest.json`

### 2. **Text Processing**
- **Chunking**: Header-aware splitting (max 700 tokens, 120 token overlap)
- **Embedding**: `e5-small-v2` with `"passage: "` prefix
- **Storage**: ChromaDB with metadata (source, title, URL)

### 3. **Hybrid Retrieval**
1. **Dense Search**: Semantic similarity using embeddings
2. **Sparse Search**: BM25 keyword matching
3. **Fusion**: Reciprocal Rank Fusion (RRF) combines results
4. **Re-ranking**: Cross-encoder scores top 20 candidates
5. **Deduplication**: Returns best chunk per source page

### 4. **User Interface**
- Chat-style input for questions
- Displays top 5 relevant results
- Expandable cards with content previews
- Links to original documentation

## 📊 Configuration

### Environment Variables
```bash
# Embedding model (default: intfloat/e5-small-v2)
EMB_MODEL=intfloat/e5-small-v2

# ChromaDB path (default: data/chroma)
CHROMA_DIR=data/chroma

# Collection name (default: fastapi-docs)
COLLECTION_NAME=fastapi-docs
```

### Retrieval Settings
Edit `app/ui_app.py`:
```python
HybridRetriever(
    topk=5,              # Number of results to return
    use_reranker=True,   # Enable cross-encoder
    rerank_topn=20       # Candidates for re-ranking
)
```

### Scraping Configuration
Edit `ingest/config.yaml` to add/remove documentation URLs.

## 🧪 Evaluation

### Generate Ground Truth Dataset
```bash
cd eval
python build_ground_truth.py --min-ans-chars 60 --max-per-file 6
```

### Compare Embedding Models
```bash
python eval/compare_embeddings.py
```

### Inspect Vector Database
```bash
python eval/inspect_chroma.py
```

## 📈 Current Status

**Completion**: ~80%

**Working**:
- ✅ Full ingestion pipeline
- ✅ Hybrid retrieval system
- ✅ Streamlit UI with result display
- ✅ Evaluation framework
- ✅ Docker deployment

**Pending**:
- ⏳ LLM integration for answer generation
- ⏳ Citation-based responses
- ⏳ Chat history and multi-turn conversations
- ⏳ Comprehensive evaluation metrics

## 🎯 Next Steps

1. **Integrate LLM** (OpenAI GPT-4 / Anthropic Claude / Ollama)
   - Implement prompt template with citations
   - Add streaming responses
   - Handle API errors

2. **Enhance Evaluation**
   - Expand ground truth dataset (100+ Q&A pairs)
   - Benchmark retrieval metrics (Hit Rate, MRR, NDCG)
   - Optimize chunk size and overlap

3. **Production Features**
   - User feedback mechanism (thumbs up/down)
   - Query logging and analytics
   - Performance monitoring

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- LLM provider integrations
- Additional evaluation metrics
- UI/UX enhancements
- Documentation expansion

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- FastAPI documentation: https://fastapi.tiangolo.com
- Sentence-Transformers by HuggingFace
- ChromaDB for vector storage
- Streamlit for rapid UI development

---

**Built with ❤️ for the FastAPI community**
