# src/ingestion/worker.py

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer


async def ingest_document(file_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Ingest a document by:
      1. Reading the file
      2. Chunking the text
      3. Embedding each chunk
      4. Indexing the embedded chunks

    Returns:
        {
            "status": "success",
            "file_path": "...",
            "chunks": N
        }
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")

    chunks = chunk_text(text)

    indexed = 0

    for i, chunk in enumerate(chunks):
        embedding = await Embedder.embed(chunk)

        payload = {
            "text": chunk,
            "embedding": embedding,
            "metadata": {
                **metadata,
                "file_path": str(path),
                "chunk_index": i,
            },
        }

        result = Indexer.index(payload)
        if asyncio.iscoroutine(result):
            await result

        indexed += 1

    return {
        "status": "success",
        "file_path": str(path),
        "chunks": indexed,
    }


class WorkerSettings:
    """
    arq worker configuration.
    """

    functions = [ingest_document]

    redis_settings = RedisSettings()

    job_timeout = 60 * 30  
    max_jobs = 10