# src/ingestion/worker.py
from __future__ import annotations

import os
import asyncio
import json 
from pathlib import Path
from typing import Any
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

from arq import create_pool
from arq.connections import RedisSettings

from src.ingestion.chunker import chunk_text, Chunk
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.config.config import CONFIG
from src.ingestion.data_pipeline import seed_postgres

import psycopg2

# Initialize embedder once (reused across jobs)
embedder = Embedder()

# Extract text based on file extension
def extract_text(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() for page in reader.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".html" or ext == ".htm":
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text()
    else:
        # Assume plain text
        return file_path.read_text(encoding="utf-8")


async def update_job_status(
    job_id: str,
    status: dict[str, Any],
):
    redis = await create_pool(RedisSettings())
    await redis.set(
        f"job:status:{job_id}",
        json.dumps(status),
    )
    await redis.close()


async def ingest_document(
    ctx,
    file_path: str,
    metadata: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:

    await update_job_status(
        job_id,
        {
            "status": "processing",
            "result": None,
        },
    )

    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # 1. Extract text (for now, simple text extraction)
        text = extract_text(path)

        # 2. Chunk the text
        chunks = chunk_text(text, metadata=metadata)
        

        # 3. Extract chunk texts for embedding
        chunk_texts = [c.text for c in chunks]

        # 4. Embed all chunks
        embeddings = embedder.embed(chunk_texts)

        # 5. Index into Qdrant
        indexer = Indexer(config=CONFIG)
        indexer.ensure_collection()
        indexer.index(chunks, embeddings)
        
        conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag_metadata"),
        user=os.getenv("POSTGRES_USER", "raglab"),
        password=os.getenv("POSTGRES_PASSWORD", "raglab"),)

        conn.autocommit = True
        seed_postgres(conn, chunk_texts)
        
        result = {
            "file_path": str(path),
            "chunks": len(chunks),
            "metadata": metadata,
        }

        await update_job_status(
            job_id,
            {
                "status": "done",
                "result": result,
            },
        )

        return result

    except Exception as e:
        await update_job_status(
            job_id,
            {
                "status": "failed",
                "result": {
                    "error": str(e)
                },
            },
        )
        raise


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings()
    job_timeout = 60 * 30  # 30 minutes
    max_jobs = 10
    max_tries = 3