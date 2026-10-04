# RAG Benchmark Studio

An evaluation-driven engineering platform for systematically analyzing, benchmarking, and optimizing Retrieval-Augmented Generation (RAG) pipelines across Dense, BM25, Hybrid, and Cross-Encoder retrieval strategies.

---

## 1. Project Overview

Most RAG implementations begin and end as simple chatbot wrappers around a single vector database and an LLM. However, in production retrieval systems:
- Semantic vector search often fails on precise technical identifiers, keywords, or acronyms.
- BM25 keyword search fails when users ask questions using synonyms or semantic paraphrasing.
- Naive hybrid search merges candidate pools but struggles with optimal score calibration.
- Multi-stage retrieval with cross-encoder reranking provides substantial ranking improvements at the cost of noticeable latency overhead.

**RAG Benchmark Studio** addresses these core trade-offs. Rather than treating RAG as a black box, it provides a full evaluation and experimentation platform where engineers can:
1. Compare retrieval quality (**Recall@K**, **Hit Rate@K**, **MRR**) and runtime latency across 4 distinct retrieval strategies on the same document corpus.
2. Inspect exact retrieved context passages and source/page metadata.
3. Generate grounded answers with citations using Google Gemini.
4. Run controlled experiments evaluating how chunk size and Top-K values impact retrieval performance and latency.

---

## 2. Main Features

- **Multi-Document PDF Ingestion**: PyPDF-based parser extracting text alongside file source and page metadata.
- **Recursive Character Chunking**: Configurable chunk size and overlap preserving sentence structure and page labels.
- **Dense Vector Retrieval**: Semantic search powered by `sentence-transformers/all-MiniLM-L6-v2` with FAISS inner-product indexing.
- **BM25 Keyword Retrieval**: Exact term frequency and inverse document frequency retrieval using `rank-bm25`.
- **Hybrid Search**: Convex linear combination of min-max normalized Dense and BM25 relevance scores ($\alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{bm25}}$).
- **Two-Stage Cross-Encoder Reranking**: High-precision joint query-document scoring using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **LLM Answer Generation & Citations**: Grounded answer synthesis via Google Gemini (`gemini-3.8-flash`) with automatic source and page citation formatting.
- **Independent Retrieval Benchmarking**: Automated evaluation engine computing Recall@K, Hit Rate@K, MRR, and per-query latency without calling expensive LLM APIs.
- **Lightweight Generation Metrics**: Heuristic groundedness/faithfulness, context relevance, context precision, and context recall.
- **Interactive Streamlit Studio**: Multi-tab interface featuring interactive Q&A, side-by-side retrieval comparison, retrieved context inspection, benchmark execution, and experiment visualizations.
- **FAISS Index Persistence**: Disk caching for vector indexes to eliminate cold-start embedding recomputation.

---

## 3. Architecture

```text
               PDF Documents (data/documents/*.pdf)
                                ↓
                      Document Loader (PyPDF)
                                ↓
                 Recursive Text Chunker (500 / 100)
                                ↓
            ┌───────────────────┼───────────────────┐
            ↓                   ↓                   ↓
      Dense Retriever     BM25 Retriever     Hybrid Retriever
    (all-MiniLM-L6-v2)     (rank-bm25)      (Dense + BM25 Score)
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ↓
                    Cross-Encoder Reranker
                 (ms-marco-MiniLM-L-6-v2)
                                ↓
                     Top-K Retrieved Chunks
                                ↓
                      Context & Prompt Builder
                                ↓
                        RAGGenerator
                                ↓
                    Google Gemini LLM API
                                ↓
                    Grounded Answer + Citations
```

---

## 4. Technology Stack

- **Language & Runtime**: Python 3.11
- **UI & Web Application**: Streamlit
- **Visualization**: Plotly Express & Pandas
- **Vector Search & Embeddings**: FAISS (`faiss-cpu`), `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Keyword Search**: `rank-bm25`
- **Document Processing**: `pypdf`, `langchain-text-splitters`
- **LLM Integration**: `langchain-google-genai` (Google Gemini), `langchain-openai` (optional fallback)
- **Environment & Config**: `python-dotenv`
- **Testing**: `pytest`

---

## 5. Retrieval Methods

| Strategy | How It Works | Primary Strength | Primary Limitation |
| :--- | :--- | :--- | :--- |
| **Dense Retrieval** | Bi-encoder converts query and chunks to 384-d normalized embeddings; FAISS computes cosine similarity via inner product. | Captures semantic intent and conceptual synonyms even when query terms do not appear verbatim. | Misses exact keyword matches, domain acronyms, or specific alphanumeric codes. |
| **BM25 Retrieval** | Inverted index tokenization scoring relevance based on term frequency and document length saturation. | Extremely fast (<2 ms), deterministic, and exact on specialized keywords or product names. | Vocabulary mismatch problem; incapable of semantic generalization. |
| **Hybrid Retrieval** | Retrieves top candidates from both Dense and BM25, normalizes scores using min-max scaling, and linearly blends them ($\alpha=0.5$). | Combines semantic understanding with exact lexical matching for robust general retrieval. | Requires score normalization calibration across disparate score distributions. |
| **Hybrid + Cross-Encoder** | Hybrid retriever fetches an initial candidate pool (Top 10), which is then re-scored by a joint query-document cross-encoder. | Maximum precision and ranking quality; captures deep cross-attention token interactions. | Higher computational latency (~300–400 ms CPU inference vs <20 ms for first-stage search). |

---

## 6. RAG Generation Pipeline

```text
Query ──► Retriever ──► Top-K Chunks ──► Build Context ──► Build Prompt ──► Gemini ──► Answer + Citations
```

1. **Context Construction**: Formatted chunks include source file name and page number:
   `[Source 1: attention_is_all_you_need.pdf, page 3] ...text...`
2. **Prompt Engineering**: The prompt strictly instructs the model to rely solely on the provided context. If the answer cannot be determined from the passages, the model explicitly responds with that fact.
3. **Citation Preservation**: Document metadata (`source`, `page_label`) is parsed into structured citations (`attention_paper.pdf, page 1`) displayed directly below the generated answer.

---

## 7. Retrieval & Generation Evaluation

### Retrieval Metrics (LLM-Independent)
- **Recall@K**: Proportion of queries where at least one ground-truth source was retrieved within Top-K.
- **Hit Rate@K**: Fraction of queries with at least one relevant document in the top K results.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant document across queries ($\frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$).
- **Retrieval Latency**: End-to-end execution time in milliseconds per retrieval strategy.

### Generation Evaluation (Heuristic / Dependency-Free)
The project implements lightweight, deterministic generation metrics that do not depend on external evaluation SaaS or fragile heavy dependencies:
- **Faithfulness / Groundedness**: Token-overlap and sentence-level containment verifying answer claims exist in context.
- **Context Relevance**: Semantic and lexical relevance of retrieved context chunks to the user query.
- **Context Precision**: Ratio of relevant chunks positioned at top ranks in the retrieved context.
- **Context Recall**: Coverage of ground-truth reference statements captured by the retrieved context.

---

## 8. Benchmark Observations

Measured performance on the seminal AI papers corpus (564 chunks, 4 ground-truth benchmark queries):

| Pipeline | Recall@5 | Hit Rate@5 | MRR | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **BM25** | 0.75 | 0.75 | 0.5625 | ~1.3 ms |
| **Dense** | 1.00 | 1.00 | 0.8125 | ~18.3 ms |
| **Hybrid** | 1.00 | 1.00 | 1.0000 | ~17.4 ms |
| **Hybrid + Cross-Encoder** | 1.00 | 1.00 | 1.0000 | ~376 ms |

### Key Trade-Off Insights
- **Hybrid Retrieval** achieves optimal ranking (MRR = 1.00) while keeping latency under 20 ms.
- **Cross-Encoder Reranking** provides optimal precision on ambiguous candidate pools but adds ~350 ms of CPU inference overhead.
- **BM25** is 14x faster than Dense search but drops recall on queries requiring semantic abstraction.

---

## 9. Experiments & Analysis

The **Experiments & Analysis** tab in the Streamlit application provides automated parameter sweeps:
1. **MRR vs Latency Trade-Off**: Scatter visualization mapping retrieval effectiveness against execution speed.
2. **Top-K Sensitivity Sweep**: Evaluates retrieval performance across $K \in [1, 10]$ to identify the optimal context window size.
3. **Chunk Size Impact**: Compares retrieval metrics across chunk configurations (e.g., 250, 500, 1000 tokens) to assess context granularity trade-offs.

---

## 10. Local Setup & Installation

### Prerequisites
- Python 3.11 installed (verified on Windows and Linux).
- Git.

### 1. Clone Repository & Setup Virtual Environment
```powershell
git clone https://github.com/Jenish045/rag-benchmark-studio.git
cd rag-benchmark-studio

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Edit `.env` to supply your Google Gemini API key:
```env
LLM_API_KEY=AIzaSy...your_gemini_api_key
LLM_MODEL=gemini-3.8-flash
APP_ENV=development
DEFAULT_TOP_K=5
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=100
```
> **Security Notice**: Never commit `.env` or paste API keys into Git. `.env` is ignored in `.gitignore`.

### 4. Run Automated Test Suite
```powershell
py -3.11 -m pytest --basetemp=.pytest_temp -q
```
*Expected: 122 tests passed.*

### 5. Launch Streamlit Application
```powershell
py -3.11 -m streamlit run app/main.py
```
Open your browser to `http://localhost:8501`.

---

## 11. Project Structure

```text
RAG_Benchmark_Studio/
├── .streamlit/
│   ├── config.toml               # Streamlit server & UI configuration
│   └── secrets.toml.example      # Example secrets template for cloud deployment
├── app/
│   └── main.py                   # Streamlit studio application (5 tabs + sidebar)
├── data/
│   ├── documents/                # Source PDF documents
│   ├── evaluation/               # Benchmark query datasets
│   └── indexes/                  # Persisted FAISS vector indexes (gitignored)
├── src/
│   └── rag_benchmark/
│       ├── benchmarking/         # Benchmark engine, experiments & analysis
│       ├── chunking/             # Text chunking strategies
│       ├── evaluation/           # Retrieval and generation metrics
│       ├── generation/           # RAGGenerator, context, prompt builders
│       ├── ingestion/            # PDF and document loaders
│       ├── reranking/            # Cross-encoder reranking
│       ├── retrieval/            # Dense, BM25, and Hybrid retrievers
│       └── utils/                # Config and environment loaders
├── tests/                        # 121 comprehensive unit & integration tests
├── .env.example                  # Environment variable template
├── .gitignore                    # Git exclusion rules
├── pyproject.toml                # Project packaging metadata
├── README.md                     # Project documentation
└── requirements.txt              # Production and testing dependencies
```

---

## 12. Deployment & Packaging (Streamlit Cloud)

The application is fully prepared for zero-configuration deployment on **Streamlit Community Cloud**:

### Deployment Steps:
1. Push the repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Select your repository and configure:
   - **Main file path**: `app/main.py`
   - **Python version**: `3.11`
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   LLM_API_KEY = "your-google-gemini-api-key"
   LLM_MODEL = "gemini-3.8-flash"
   ```
5. Click **Deploy**.

### Ephemeral Storage Resilience
Streamlit Cloud instances feature ephemeral filesystems. Vector index persistence in `DenseRetriever` automatically catches read/write exceptions and falls back to in-memory index construction if the disk is read-only or wiped between container restarts.

---

## 13. System Limitations

- **Benchmark Query Scope**: Benchmark evaluation currently includes 4 representative queries covering the 4 foundational AI papers.
- **Free-Tier LLM Rate Limits**: Google Gemini free-tier keys are subject to per-minute request (RPM) and token (TPM) limits.
- **Generation Metric Heuristics**: Generation quality evaluation uses token-overlap and fuzzy alignment metrics rather than a paid LLM-as-a-judge model.
- **Index Invalidation**: FAISS cache invalidation verifies total document/chunk counts. Modifying chunk content without changing chunk count requires deleting `data/indexes/dense_faiss.index`.

---

## 14. Future Improvements

- Integrate synthetic test dataset generation for automated domain adaptation.
- Add support for dynamic reranking thresholds based on first-stage score margins.
- Provide GPU acceleration options for dense embedding and cross-encoder inference in high-throughput environments.
- Expand generation evaluation with optional local GGUF / Ollama judge models.

---

## 15. Application Views

When running `app/main.py`, the following 5 views are available:
1. **Chat Tab**: Interactive question-answering with retriever selection, latency display, answer synthesis, and source citations.
2. **Retrieval Comparison Tab**: Side-by-side card comparison of all 4 retrieval strategies for the active query.
3. **Retrieved Context Tab**: Expandable inspection of raw text chunks and page metadata.
4. **Benchmark Tab**: One-click benchmark execution rendering the comparative metrics table and MRR vs Latency chart.
5. **Experiments & Analysis Tab**: Automated parameter sweeps exploring Top-K sensitivity and chunk size trade-offs.