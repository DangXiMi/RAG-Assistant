# src/ingestion/worker.py
from __future__ import annotations

import os
import asyncio
import json 
import logging
from pathlib import Path
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from src.ingestion.chunker import chunk_text, chunk_pages, Chunk
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.config.config import CONFIG
from src.ingestion.data_pipeline import seed_postgres

import psycopg2

from src.utils.helper_func import load_pages

from src.ingestion.deduplicated import Deduplicator


embedder = Embedder()
logger = logging.getLogger(__name__)
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))


async def update_job_status(
    job_id: str,
    status: dict[str, Any],
    redis,
):
    #redis = await create_pool(RedisSettings())
    await redis.set(
        f"job:status:{job_id}",
        json.dumps(status),
    )
    #await redis.close()


async def ingest_document(
    ctx,
    file_path: str,
    metadata: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    redis = await create_pool(RedisSettings(host=redis_host, port=redis_port))

    await update_job_status(
        job_id,
        {
            "status": "processing",
            "result": None,
        },
        redis,
    )

    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # 1. Load the document into page/sheet-scoped units, so that page
        #    numbers and sheet names survive into the chunk metadata and can
        #    be cited. OCR runs here for images and scanned PDF pages.
        pages = load_pages(path, metadata=metadata)

        if not pages:
            raise ValueError(
                f"No extractable content found in {path.name}"
            )

        # 2. Chunk per unit, keeping each unit's location metadata
        chunks = chunk_pages(pages, metadata=metadata)

        if not chunks:
            raise ValueError(
                f"No text chunks produced from {path.name}"
            )

        logger.info(
            "Extracted %d unit(s) and %d chunk(s) from %s",
            len(pages),
            len(chunks),
            path.name,
        )

        # 3. Extract chunk texts and check deduplicate for embedding
        dedup = Deduplicator(redis)
        filtered_chunks = []
        chunk_texts = []
        duplicates = 0

        for chunk in chunks:
            if dedup.enabled:
                is_new = await dedup.add_if_new(chunk.text)
                if not is_new:
                    logger.info("Duplicate skipped: %s...", chunk.text[:50])
                    duplicates += 1
                    continue
            filtered_chunks.append(chunk)
            chunk_texts.append(chunk.text)

        # Everything was already indexed (a re-upload of identical content, or a
        # partially overlapping document). This is a successful no-op, not a
        # failure: embedding an empty list and upserting nothing makes Qdrant
        # reject the request with "Empty update request".
        if not filtered_chunks:
            result = {
                "file_path": str(path),
                "chunks": 0,
                "chunks_extracted": len(chunks),
                "chunks_skipped_as_duplicate": duplicates,
                "pages": len(pages),
                "ocr_units": sum(1 for p in pages if p.extraction == "ocr"),
                "metadata": metadata,
                "note": (
                    "All chunks were already indexed; nothing new to add. "
                    "Remove the dedup keys in Redis to force re-indexing."
                ),
            }

            logger.info(
                "Skipped %s: all %d chunk(s) were already indexed",
                path.name,
                len(chunks),
            )

            await update_job_status(
                job_id,
                {
                    "status": "done",
                    "result": result,
                },
                redis,
            )
            await redis.close()

            return result

        # 4. Embed all chunks
        embeddings = embedder.embed(chunk_texts)

        # 5. Index into Qdrant
        #    Must use filtered_chunks: the filtered list is what the embeddings
        #    correspond to, so passing `chunks` here would misalign them.
        indexer = Indexer(config=CONFIG)
        indexer.ensure_collection()
        indexer.index(filtered_chunks, embeddings)
        
        # 6. Index into PostgreSQL (sparse)
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
            "chunks": len(filtered_chunks),
            "chunks_extracted": len(chunks),
            "pages": len(pages),
            "ocr_units": sum(1 for p in pages if p.extraction == "ocr"),
            "metadata": metadata,
        }

        await update_job_status(
            job_id,
            {
                "status": "done",
                "result": result,
            },
            redis,
        )
        await redis.close()

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
            redis,
        )
        raise


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings(host=redis_host, port=redis_port)
    job_timeout = 60 * 30  # 30 minutes
    max_jobs = 10
    max_tries = 3