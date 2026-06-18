# RAG-KNOWLEDGE-ASSISTANT
`rag-knowledge-assistant` ingests a knowledge base, indexes them, and answers user questions with exact citations. It supports multiple retrieval strategies, automatically evaluates answer quality, and runs entirely via Docker.

**Real‑world usage scenarios**
- Internal company wiki Q&A
- Customer support bot that cites documentation
- Research assistant that answers from papers
- Compliance chatbot answering policy questions with traceable sources

## Project Structure
```
production-rag/
├── .github/workflows/          # CI (lint, test, eval)
├── src/
│   ├── ingestion/              # chunking, embedding, storing
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── indexer.py
│   ├── retrieval/              # dense, sparse, hybrid, routing
│   │   ├── base.py             # abstract BaseRetriever
│   │   ├── dense_qdrant.py
│   │   ├── sparse_bm25.py
│   │   ├── hybrid_rrf.py
│   │   └── router.py           # decides to use RAG or not
│   ├── reranking/              # cross-encoder
│   │   └── reranker.py
│   ├── generation/             # prompt + LLM + citations
│   │   ├── prompt_builder.py
│   │   ├── llm_client.py
│   │   └── citation_extractor.py
│   ├── evaluation/             # RAGAS, golden dataset
│   │   ├── metrics.py
│   │   └── runner.py
│   ├── observability/          # logging, tracing (LangSmith wrapper)
│   │   └── tracer.py
│   └── api/                    # FastAPI endpoints
│       ├── main.py
│       ├── routes/query.py
│       └── routes/ingest.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── eval/                   # regression tests
├── infra/
│   ├── docker-compose.yml
│   └── Dockerfile              # main Python service
├── config/
│   ├── default.yaml            # model names, batch sizes
│   └── production.yaml
├── scripts/
│   └── init_qdrant_collection.py
├── .env.example
└── README.md
```
src/ – installable package (pip install -e .). Clean imports.

ingestion/ – isolated from query path. Can be run as a separate process (CLI or background worker).

retrieval/base.py – defines retrieve(query, top_k). All retrievers implement it. API code only depends on BaseRetriever.

evaluation/ – from day 1, even dummy, to enforce thinking about metrics.

infra/docker-compose.yml – Qdrant, Postgres, Redis. One command to start all backing services.

config/ – no hardcoded constants. Change embedding model without touching code.

## Installation

Install pip-tools:

```bash
pip install pip-tools
```

Sync dependencies from the lock file:

```bash
pip-sync requirements.lock
```

To regenerate the lock file after updating `requirements.in`:

```bash
pip-compile --generate-hashes -o requirements.lock requirements.in
```