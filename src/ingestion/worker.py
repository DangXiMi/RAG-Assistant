# src/ingestion/worker.py
from __future__ import annotations

import asyncio
import json 
from pathlib import Path
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from src.ingestion.chunker import chunk_text, Chunk
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer


# Initialize embedder once (reused across jobs)
embedder = Embedder()


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
        text = path.read_text(encoding="utf-8")

        # 2. Chunk the text
        chunks = chunk_text(text, metadata=metadata)

        # 3. Extract chunk texts for embedding
        chunk_texts = [c.text for c in chunks]

        # 4. Embed all chunks
        embeddings = embedder.embed(chunk_texts)

        # 5. Index into Qdrant
        indexer = Indexer()
        indexer.ensure_collection()
        indexer.index(chunks, embeddings)

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