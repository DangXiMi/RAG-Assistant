"""Shared corpus seeding for the evaluation and demo scripts.

Both `scripts/test_e2e.py` and `scripts/run_ragas_evaluation.py` used to call a
local `seed_postgres()` that ran `DROP TABLE chunks CASCADE` and minted throwaway
`uuid4()` ids. Two consequences:

* running any evaluation destroyed whatever else was indexed, including the demo
  corpus, leaving hybrid retrieval with an empty keyword half;
* the sparse rows carried ids that no dense vector shared, so RRF could not fuse
  the two stores.

This module seeds both stores through the real ingestion components instead, so
ids match and nothing is destroyed. It uses a dedicated Qdrant collection by
default, so an evaluation run cannot disturb the demo collection.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import CONFIG  # noqa: E402
from src.ingestion import sparse_store  # noqa: E402
from src.ingestion.chunker import chunk_pages  # noqa: E402
from src.ingestion.embedder import Embedder  # noqa: E402
from src.ingestion.indexer import Indexer  # noqa: E402
from src.ingestion.loaders.interface import LoadedPage, base_metadata  # noqa: E402

logger = logging.getLogger(__name__)

# Evaluation corpora live in their own collection, so the demo is untouched.
DEFAULT_COLLECTION = os.getenv("EVAL_QDRANT_COLLECTION", "rag_eval")


def build_chunks(documents, source_name: str = "sample.txt", page_metadata: bool = True):
    """Turn raw document strings into chunks tagged with citation metadata.

    Args:
        documents: Iterable of document strings.
        source_name: Value recorded as `source`, used in citations.
        page_metadata: Tag each document with a page number so citations show
            one, matching how the loaders behave for real files.
    """
    path = Path(source_name)
    pages = [
        LoadedPage(
            text=text,
            index=index,
            metadata={
                **base_metadata(path),
                "page": index + 1,
                "page_count": len(documents),
            },
        )
        for index, text in enumerate(documents)
    ]
    if not page_metadata:
        for page in pages:
            page.metadata.pop("page", None)
            page.metadata.pop("page_count", None)

    return chunk_pages(pages)


def replace_source(
    documents,
    source_name: str,
    collection_name: str = DEFAULT_COLLECTION,
    embedder: Embedder | None = None,
):
    """Seed an evaluation corpus without disturbing anything else.

    Only rows belonging to `source_name` are removed from the shared sparse
    table. The Qdrant collection is dedicated to evaluation, so it is reset
    wholesale.

    This matters because the sparse table is shared by every corpus: clearing
    it entirely would delete the demo's chunks and leave the demo with dense
    retrieval only.
    """
    conn = sparse_store.connect()
    try:
        sparse_store.ensure_table(conn)
        removed = sparse_store.delete_source(conn, source_name)
        if removed:
            logger.info("Removed %d stale chunk(s) for %r", removed, source_name)
    finally:
        conn.close()

    # Reset the dedicated evaluation collection so re-runs are deterministic.
    from qdrant_client import QdrantClient

    client = QdrantClient(host="localhost", port=6333)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.close()

    return seed(
        documents,
        source_name=source_name,
        collection_name=collection_name,
        embedder=embedder,
    )


def seed(
    documents,
    source_name: str = "sample.txt",
    collection_name: str = DEFAULT_COLLECTION,
    reset_sparse: bool = False,
    embedder: Embedder | None = None,
):
    """Embed and index `documents` into both Qdrant and the sparse store.

    Args:
        documents: Document strings to index.
        source_name: Citation source name.
        collection_name: Qdrant collection to write to.
        reset_sparse: Clear the *whole* sparse index first. Defaults to False,
            because the table is shared by every corpus and clearing it would
            silently strip other corpora of their keyword index. Use
            `replace_source` to refresh just one corpus.
        embedder: Reuse an existing Embedder to avoid reloading the model.

    Returns:
        `(chunks, indexer, embedder)`.
    """
    chunks = build_chunks(documents, source_name=source_name)
    logger.info("Built %d chunk(s) from %d document(s)", len(chunks), len(documents))

    if embedder is None:
        embedder = Embedder()

    indexer = Indexer(config={**CONFIG, "qdrant": {**CONFIG["qdrant"], "collection_name": collection_name}})
    indexer.ensure_collection()

    vectors = embedder.embed([chunk.text for chunk in chunks])
    indexer.index(chunks, vectors)

    conn = sparse_store.connect()
    try:
        sparse_store.ensure_table(conn)
        if reset_sparse:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chunks")
        sparse_store.upsert_chunks(conn, chunks)
    finally:
        conn.close()

    logger.info(
        "Seeded %d chunk(s) into Qdrant '%s' and the sparse store",
        len(chunks),
        collection_name,
    )
    return chunks, indexer, embedder
