# RAG-KNOWLEDGE-ASSISTANT

`rag-knowledge-assistant` ingests a knowledge base, indexes it, and answers user questions with exact citations. It supports multiple retrieval strategies, automatically evaluates answer quality, and runs entirely via Docker.

**Real‑world usage scenarios**
- Internal company wiki Q&A
- Customer support bot that cites documentation
- Research assistant that answers from papers
- Compliance chatbot answering policy questions with traceable sources

---
## Tech Stack

Backend
- FastAPI

Vector Database
- Qdrant

Metadata
- PostgreSQL

Queue
- Redis + ARQ

Embedding
- Sentence Transformers

LLM
- Ollama

Frontend
- Streamlit

Evaluation
- RAGAS

---

## 🚀 Quick Start (One Command)

```bash
docker-compose up --build
```

This starts:
- **Qdrant** (vector DB)
- **PostgreSQL** (metadata + FTS)
- **Redis** (job queue)
- **Ollama** (local LLM)
- **FastAPI** (backend)
- **arq Worker** (async ingestion)
- **Streamlit UI** (frontend)

Then open:
- **UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/v1/health → `{"status":"ok"}`

---

## ✨ Features

- **Hybrid Retrieval** – Combines dense (Qdrant) and sparse (PostgreSQL FTS) search using Reciprocal Rank Fusion (RRF).
- **Advanced Query Rewriting** – HyDE (Hypothetical Document Embeddings) and Multi-Query expansion.
- **Cross-Encoder Reranking** – Re-ranks top candidates with a `ms-marco-MiniLM-L-6-v2` cross‑encoder for higher precision.
- **Local LLM Integration** – Uses Ollama (llama3.1:8b) for fully offline, privacy‑friendly generation.
- **Citation‑Grounded Answers** – Every answer includes source document IDs, and the prompt enforces “I don’t know” for out‑of‑context queries.
- **Async Ingestion** – Upload documents and let the background worker process them without blocking the UI.
- **Deduplication & Quality Scoring** – Prevents duplicate chunks and filters low‑quality content.
- **Evaluation Suite** – RAGAS metrics (faithfulness, answer relevancy, context precision, recall) with a dedicated dashboard.
- **LangSmith Observability** – Full tracing of retrieval and generation steps.
- **Modern Streamlit UI** – Chat interface with file upload, real‑time job status, and source citation display.

---

## 🗂️ Project Structure

```
rag-knowledge-assistant/
├── .env.example                          # Environment variables template
├── .python-version                       # Python 3.11
├── requirements.in                       # Loose dependencies
├── requirements.lock                     # Pinned dependencies with hashes
├── README.md                             # This file
├── docker-compose.yml                    # Orchestrates all services
├── infras/
│   ├── Dockerfile.api                        # API container
│   ├── Dockerfile.worker                     # Worker container
│   └── Dockerfile.ui                         # UI container
│
├── src/
│   ├── config/
│   │   ├── config.py                     # Loads YAML and exports CONFIG constants
│   │   └── config.yaml                   # Central configuration
│   ├── ingestion/
│   │   ├── chunker.py                    # Sentence-aware chunking with overlap
│   │   ├── data_pipeline.py              # load_pipeline() for the API
│   │   ├── deduplicator.py               # Exact hash deduplication (Redis)
│   │   ├── embedder.py                   # SentenceTransformer (all-MiniLM-L6-v2)
│   │   ├── indexer.py                    # Qdrant vector indexer
│   │   └── worker.py                     # arq worker for async ingestion
│   ├── retrieval/
│   │   ├── dense_retriever.py            # Qdrant vector search
│   │   ├── sparse_retriever.py           # PostgreSQL full-text search (ts_rank)
│   │   ├── hybrid_retriever.py           # RRF fusion of dense + sparse
│   │   ├── hyde_retriever.py             # Hypothetical Document Embeddings
│   │   └── multi_query_retriever.py      # Multi-query expansion with RRF
│   ├── reranking/
│   │   └── cross_encoder_reranker.py     # Cross-encoder reranking
│   ├── generation/
│   │   └── generator.py                  # Prompt builder + Ollama LLM caller
│   ├── evaluation/
│   │   └── ragas_evaluator.py            # RAGAS evaluation harness
│   ├── api/
│   │   ├── main.py                       # FastAPI application entrypoint
│   │   ├── models.py                     # Pydantic request/response models
│   │   └── routes/
│   │       ├── query.py                  # /api/v1/query endpoint
│   │       ├── ingest.py                 # /api/v1/ingest endpoint
│   │       └── status.py                 # /api/v1/job/{job_id} endpoint
│   ├── utils/
│   │   └── rrf.py                        # RRF fusion helper
│   └── ui/
│       ├── app.py                        # Streamlit frontend (calls FastAPI)
│       └── app_dashboard.py              # Streamlit evaluation dashboard
│
├── scripts/
│   ├── test_e2e.py                       # End-to-end smoke test
│   ├── run_evaluation.py                 # Runs RAGAS evaluation for all modes
│   └── start_worker.sh                   # Starts the arq worker
│
├── tests/
│   ├── unit/                             # Unit tests
│   └── integration/                      # Integration tests
│
├── data/
│   └── evaluation/
│       ├── golden.jsonl                  # Golden dataset for RAGAS
│       └── metrics.csv                   # Evaluation results
│
└── uploads/                              # Temporary upload directory
    └── .gitkeep
```

---

## 🚀 Installation (Manual – Without Docker)

If you prefer to run without Docker:

**1. Clone the repository**
```bash
git clone https://github.com/DangXiMi/RAG-Assistant
cd rag-knowledge-assistant
```

**2. Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install pip-tools and sync dependencies**
```bash
pip install pip-tools
pip-sync requirements.lock
```

To update dependencies after editing requirements.in:
```bash
pip-compile --generate-hashes -o requirements.lock requirements.in
pip-sync requirements.lock
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your PostgreSQL, Qdrant, and LangSmith credentials.
```

**5. Start backing services**
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name postgres -p 5432:5432 -e POSTGRES_USER=raglab -e POSTGRES_PASSWORD=raglab -e POSTGRES_DB=rag_metadata postgres:15
docker run -d --name redis -p 6379:6379 redis:7
docker run -d --name ollama -p 11434:11434 ollama/ollama
ollama pull llama3.1:8b
```

**6. Start the API and worker**
```bash
uvicorn src.api.main:app --reload &
arq src.ingestion.worker.WorkerSettings &
```

**7. Start the UI**
```bash
streamlit run src/ui/app.py
```

---

## 🧪 Usage

### 1. Upload a Document
1. Launch the Streamlit application at `http://localhost:8501`
2. In the sidebar, click **Choose a file**
3. Upload one of the supported document formats:
   - PDF (`.pdf`)
   - Microsoft Word (`.docx`)
   - HTML (`.html`)
   - Plain Text (`.txt`)
4. Click **Process Document**
5. The application will display the indexing progress in real time:
   - **Queued** → **Processing** → **Done**
6. Once processing is complete, the total number of indexed chunks will be shown.

### 2. Ask Questions
1. Enter your question in the chat input at the bottom of the page.
2. Press **Enter**.
3. The application will:
   - Retrieve the most relevant document chunks.
   - Generate an answer using the retrieved context.
   - Display the answer together with clickable source IDs for reference.

### 3. Choose a Retrieval Strategy

> **Note:** Retrieval mode selection is currently configured through the backend. A sidebar selector will be available in a future release.

| Strategy | Description |
|----------|-------------|
| **Hybrid** *(Default)* | Combines dense and sparse retrieval using Reciprocal Rank Fusion (RRF). |
| **HyDE** | Generates a hypothetical document from the query before retrieval to improve semantic matching. |
| **Multi-Query** | Creates multiple query variations and merges the results using RRF. |
| **Reranked** | Retrieves candidate documents and reranks them using a cross-encoder for improved relevance. |

---

### 🧠 Retrieval Pipeline (Under the Hood)

1. **Dense Retrieval** – `all-MiniLM-L6-v2` embeddings stored in Qdrant.
2. **Sparse Retrieval** – PostgreSQL `tsvector` with GIN index, using `ts_rank` (TF‑IDF variant).
3. **Hybrid Fusion** – RRF (`k=60`) combines dense and sparse ranks.
4. **Advanced (optional)** – HyDE / Multi‑Query rewrites the query before retrieval.
5. **Reranking (optional)** – Cross‑encoder re‑ranks the top 50 candidates.
6. **Generation** – Ollama (llama3.1:8b) generates the final answer with cited sources.

---
## 📊 RAGAS Evaluation

The retrieval strategies were evaluated using the **RAGAS** framework on the project's golden evaluation dataset.
### Metrics

| Metric | Description |
|---------|-------------|
| **Faithfulness** | Measures whether the generated answer is supported by the retrieved context. |
| **Answer Relevancy** | Measures how well the answer addresses the user's question. |
| **Context Precision** | Measures how much of the retrieved context is actually useful. |
| **Context Recall** | Measures whether the retriever found all necessary supporting information. |

### Results

| Retrieval Strategy | Faithfulness  | Answer Relevancy  | Context Precision  | Context Recall  |
|--------------------|---------------:|-------------------:|--------------------:|-----------------:|
| **Hybrid** *(Default)* | **0.90** | 0.346 | **0.90** | **0.90** |
| **HyDE** | 0.80 | 0.346 | 0.90 | **0.90** |
| **Multi-Query** | 0.80 | 0.346 | 0.80 | **0.90** |
| **Reranked** | **0.90** | 0.346 | **0.90** | **0.90** |

### Reproducing the Benchmark

```bash
python scripts/run_evaluation.py
```

The generated metrics are saved to:

```
data/evaluation/metrics.csv
```

---

## 🧪 Testing

Run the full test suite (unit + integration):
```bash
pytest tests/
```

Run only unit tests:
```bash
pytest tests/unit/
```

Run the end‑to‑end smoke test:
```bash
python scripts/test_e2e.py
```

---

## 🖥️ Backend API (FastAPI)

The system exposes a RESTful API for programmatic access and frontend integration.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check – returns `{"status": "ok"}`. |
| `POST` | `/api/v1/query` | Submit a RAG query. |
| `POST` | `/api/v1/ingest` | Upload a document for async ingestion. |
| `GET` | `/api/v1/job/{job_id}` | Check ingestion job status. |

### Query Endpoint

**Request:**
```json
POST /api/v1/query
{
    "question": "What is the JWST launch date?",
    "user_id": "user_123",
    "mode": "Hybrid"
}
```

**Response:**
```json
{
    "answer": "The James Webb Space Telescope was launched on December 25, 2021.",
    "sources": ["ede2a4f4-1a18-4859-9624-776a85d766e5"],
    "contexts": [
        "The James Webb Space Telescope (JWST) was launched on December 25, 2021..."
    ]
}
```

### Ingestion Endpoint

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
    -F "file=@document.pdf" \
    -F 'metadata={"category":"research"}'
```

**Response:**
```json
{
    "job_id": "a1b2c3d4-...",
    "status": "queued"
}
```

### Job Status Endpoint

**Request:**
```bash
GET /api/v1/job/a1b2c3d4-...
```

**Response (Processing):**
```json
{
    "status": "processing",
    "result": null
}
```

**Response (Done):**
```json
{
    "status": "done",
    "result": {
        "file_path": "uploads/a1b2c3d4_document.pdf",
        "chunks": 42,
        "metadata": {"category": "research"}
    }
}
```

---

## ⚙️ Async Ingestion (arq + Redis)

Document ingestion runs in the background to avoid blocking the API.

### Start the Worker
```bash
# Make the script executable
chmod +x scripts/start_worker.sh

# Run the worker
./scripts/start_worker.sh

# Or run directly
arq src.ingestion.worker.WorkerSettings
```

### Worker Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `job_timeout` | 1800s (30 min) | Max time per ingestion job. |
| `max_jobs` | 10 | Concurrent jobs per worker. |
| `max_tries` | 3 | Retry failed jobs up to 3 times. |

### How It Works
1. User uploads a document → API generates a `job_id` and enqueues the task.
2. Worker picks up the job → extracts text, chunks, embeds, and indexes.
3. Status is updated in Redis → frontend can poll for completion.
4. Document is now searchable → can be retrieved via the query endpoint.

---

## 📈 Monitoring & Debugging

- **FastAPI Docs** – Visit http://localhost:8000/docs for interactive API documentation.
- **LangSmith** – View traces at https://smith.langchain.com/.
- **Redis** – Inspect job status with `redis-cli GET job:status:{job_id}`.
- **Logs** – Worker logs appear in the terminal where `arq` is running.

---

## 🔧 Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| **Unable to connect to the backend (`ConnectionError`)** | The FastAPI server is not running. | Start the API server: `uvicorn src.api.main:app --reload` |
| **Worker is not processing jobs** | Redis is not running. | Start Redis: `docker run -d -p 6379:6379 redis` |
| **Job remains in `queued` status** | The ARQ worker has not been started. | Run the worker: `arq src.ingestion.worker.WorkerSettings` |
| **`FileNotFoundError` in the worker** | The upload directory is missing or incorrectly configured. | Ensure the `uploads/` directory exists and is accessible. |
| **Qdrant version mismatch warning** | The Qdrant client and server versions differ. | This warning is generally safe to ignore, or update the client/server so their versions match. |
| **UI cannot connect to the API** | `BACKEND_URL` is configured incorrectly. | Set `BACKEND_URL=http://api:8000` when using Docker Compose, then restart the services. |

---

## ✅ Why This System Is Production‑Ready

| Aspect | Implementation |
| :--- | :--- |
| **Data Persistence** | Qdrant, PostgreSQL, and Redis data persist across container restarts via Docker volumes. |
| **Async Ingestion** | Documents are processed in the background using arq + Redis – no blocking of the API or UI. |
| **Deduplication** | Exact‑hash deduplication prevents duplicate chunks (SHA‑256 stored in Redis). |
| **Quality Scoring** | Rule‑based scoring filters low‑quality chunks before indexing. |
| **Hybrid Retrieval** | Combines dense (Qdrant) and sparse (PostgreSQL FTS) search with RRF for high recall. |
| **Observability** | Full traces via LangSmith, API docs via FastAPI, and Redis job monitoring. |
| **One‑Command Deployment** | Docker Compose starts all services with a single command. |
| **Evaluation Framework** | RAGAS metrics (faithfulness, answer relevancy, context precision, recall) measure system quality. |

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Support

- **Issues:** Please open an issue on [GitHub](https://github.com/DangXiMi/RAG-Assistant)
- **Documentation:** Refer to this README
- **API Docs:** `http://localhost:8000/docs` (when running locally)

---