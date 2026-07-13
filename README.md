# RAG-KNOWLEDGE-ASSISTANT

`rag-knowledge-assistant` ingests a knowledge base, indexes it, and answers user questions with exact citations. It supports multiple retrieval strategies, automatically evaluates answer quality, and runs entirely via Docker.

**Real‑world usage scenarios**
- Internal company wiki Q&A
- Customer support bot that cites documentation
- Research assistant that answers from papers
- Compliance chatbot answering policy questions with traceable sources

---

## ✨ Features

- **Hybrid Retrieval** – Combines dense (Qdrant) and sparse (PostgreSQL FTS) search using Reciprocal Rank Fusion (RRF).
- **Advanced Query Rewriting** – HyDE (Hypothetical Document Embeddings) and Multi-Query expansion.
- **Cross-Encoder Reranking** – Re-ranks top candidates with a `ms-marco-MiniLM-L-6-v2` cross‑encoder for higher precision.
- **Local LLM Integration** – Uses Ollama (llama3.1:8b) for fully offline, privacy‑friendly generation.
- **Citation‑Grounded Answers** – Every answer includes source document IDs, and the prompt enforces “I don’t know” for out‑of‑context queries.
- **LangSmith Observability** – Full tracing of retrieval and generation steps.
- **Streamlit UI** – Interactive interface with a toggle to switch between retrieval modes (Hybrid, HyDE, Multi-Query, Reranked).

---

## 🗂️ Project Structure

```
rag-knowledge-assistant/
├── requirements.in # Loose dependencies
├── requirements.lock # Exact pinned dependencies with hashes
├── .env.example # Environment variable template
│ 
├── src/
| ├──ui/
| | └── app.py # Streamlit UI entrypoint
│ ├── config/
| | └── config.yaml # Central configuration (chunking, models, retrieval)
│ │ └── config.py # Loads YAML and exports constants
│ ├── ingestion/ # Data preparation
│ │ ├── chunker.py # Sentence-aware chunking with overlap
│ │ ├── embedder.py # SentenceTransformer (all-MiniLM-L6-v2)
│ │ └── indexer.py # Qdrant vector indexer
│  retrieval/ # Search strategies
│ │ ├── dense_retriever.py # Qdrant vector search
│ │ ├── sparse_retriever.py # PostgreSQL full‑text search (ts_rank)
│ │ ├── hybrid_retriever.py # RRF fusion of dense + sparse
│ │ ├── hyde_retriever.py # Hypothetical Document Embeddings
│ │ ├── multi_query_retriever.py
│ ├── utils/
│ │ └── rrf.py # RRF fusion helper
│ ├── reranking/
│ │ └── cross_encoder_reranker.py
│ └── generation/
│   └── generator.py # Prompt builder + Ollama LLM caller
├── tests/
│ ├── unit/ # Unit tests for each component
│ └── integration/ # Integration tests (PostgreSQL, Qdrant)
├── scripts/
│ └── test_e2e.py # End‑to‑end smoke test script
└── infra/ # (Placeholder for future Docker Compose)
```

---

## 🚀 Installation

**1. Clone the repository**
```bash
git clone https://github.com/DangXiMi/RAG-Assistant
cd rag-knowledge-assistant
```

**2. Set up a virtual environment**

``` bash
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
**4. Configure environment variables **

``` bash
cp .env.example .env
Edit .env with your PostgreSQL, Qdrant, and LangSmith credentials.
```

**5. Start backing services (PostgreSQL & Qdrant)**
You can run them manually or use Docker (recommended):

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker run -d --name postgres -p 5432:5432 -e POSTGRES_USER=raglab -e POSTGRES_PASSWORD=raglab -e POSTGRES_DB=rag_metadata postgres:15
```
---

## 🧪 Usage

**1. Populate the database with sample data**

```bash
python scripts/test_e2e.py
```
This ingests a sample knowledge base (space telescope facts) into both Qdrant and PostgreSQL, then runs a test query to verify the pipeline.

**2. Launch the Streamlit UI**

```bash
streamlit run app.py
Open http://localhost:8501 in your browser.
```

**3. Select a retrieval mode**

In the sidebar, choose between:

- Hybrid – Dense + Sparse with RRF (default, balanced).

- HyDE – Generates a hypothetical document to improve semantic retrieval.

- Multi-Query – Generates 3 query variants, retrieves for each, and fuses results with RRF.

- Reranked – Retrieves 50 candidates with hybrid, then re‑ranks with a cross‑encoder for highest precision.

Type your question and click Submit. The UI will display:

- The LLM’s answer with citations.

- The source document IDs used.

An expandable debug panel showing the retrieved chunks and their scores.

---

### 🧠 Retrieval Pipeline (Under the Hood)

1. **Dense Retrieval** – `all-MiniLM-L6-v2` embeddings stored in Qdrant.
2. **Sparse Retrieval** – PostgreSQL `tsvector` with GIN index, using `ts_rank` (TF‑IDF variant).
3. **Hybrid Fusion** – RRF (`k=60`) combines dense and sparse ranks.
4. **Advanced (optional)** – HyDE / Multi‑Query rewrites the query before retrieval.
5. **Reranking (optional)** – Cross‑encoder re‑ranks the top 50 candidates.
6. **Generation** – Ollama (llama3.1:8b) generates the final answer with cited sources.

---

## 📊 Observability

All retrieval and generation steps are traced via **LangSmith**. Once your API key is set in `.env`, you can view traces in the LangSmith dashboard to debug, monitor latency, and evaluate retrieval quality.

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

**Response:
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
**Request:***

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
Response (Processing):
```
```json
{
    "status": "processing",
    "result": null
}
```
**Response:***

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

# Start the worker
./scripts/start_worker.sh

# Or run directly
arq src.ingestion.worker.WorkerSettings
```

### Worker Configuration

Setting	Value	Description
job_timeout	1800s (30 min)	Max time per ingestion job.
max_jobs	10	Concurrent jobs per worker.
max_tries	3	Retry failed jobs up to 3 times.

### How It Works
1. User uploads a document → API generates a job_id and enqueues the task.

2. Worker picks up the job → extracts text, chunks, embeds, and indexes.

3. Status is updated in Redis → frontend can poll for completion.

4. Document is now searchable → can be retrieved via the query endpoint.
---

## 📈 Monitoring & Debugging

FastAPI Docs – Visit http://localhost:8000/docs for interactive API documentation.

LangSmith – View traces at https://smith.langchain.com/.

Redis – Inspect job status with redis-cli GET job:status:{job_id}.

Logs – Worker logs appear in the terminal where start_worker.sh is running.

---

## 🔧 Troubleshooting
Issue	Likely Cause	Fix
ConnectionError to backend	FastAPI not running	Run python -m src.api.main
Worker not processing jobs	Redis not running	Start Redis: docker run -d -p 6379:6379 redis
Job stuck in queued	Worker not started	Run ./scripts/start_worker.sh
FileNotFoundError in worker	Upload path incorrect	Check uploads/ directory exists
text

---

## Test
test the full ingestion flow:

```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis

# 2. Start the worker
arq src.ingestion.worker.WorkerSettings

# 3. In another terminal, start FastAPI
uvicorn src.api.main:app --reload

# 4. Upload a test file
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@test.txt"

# 5. Check the job status
curl http://localhost:8000/api/v1/job/{job_id}


## 📄 License
