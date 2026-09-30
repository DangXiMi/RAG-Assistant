"""Reset the stores and seed the space-domain corpus the golden set expects.

The refund-policy documents ingested by the live end-to-end test are unrelated
to the JWST/Hubble golden questions, so evaluating against them would measure
nothing. This clears both stores and loads `SAMPLE_DOCS` from scripts/test_e2e.py
through the real loader/chunker/sparse-store path, so ids are shared.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from qdrant_client import QdrantClient

from scripts.test_e2e import SAMPLE_DOCS
from src.config.config import CONFIG
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.ingestion.loaders.interface import LoadedPage, base_metadata
from src.ingestion import sparse_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_corpus")


def reset_sparse(conn) -> None:
    sparse_store.reset_index(conn)
    logger.info("Sparse index reset")


def reset_dense() -> None:
    client = QdrantClient(
        host="localhost", port=6333
    )
    name = CONFIG["qdrant"]["collection_name"]
    if client.collection_exists(name):
        client.delete_collection(name)
        logger.info("Dropped dense collection %s", name)
    client.close()


def main() -> int:
    reset_dense()

    conn = sparse_store.connect()
    try:
        reset_sparse(conn)

        # One page per sample document, tagged so citations name the source.
        path = Path("jwst_sample.txt")
        pages = [
            LoadedPage(
                text=text,
                index=i,
                metadata={**base_metadata(path), "page": i + 1, "page_count": len(SAMPLE_DOCS)},
            )
            for i, text in enumerate(SAMPLE_DOCS)
        ]

        chunks = chunk_pages(pages)
        logger.info("Built %d chunk(s) from %d sample document(s)", len(chunks), len(pages))

        embedder = Embedder()
        indexer = Indexer(config=CONFIG)
        indexer.ensure_collection()
        vectors = embedder.embed([c.text for c in chunks])
        indexer.index(chunks, vectors)

        sparse_store.ensure_table(conn)
        sparse_store.upsert_chunks(conn, chunks)

        print()
        print("SEEDED")
        print(f"  chunks            : {len(chunks)}")
        print(f"  dense points      : {indexer.client.count(indexer.collection_name, exact=True).count}")
        print(f"  sparse rows       : {sparse_store.count_chunks(conn)}")
        print(f"  sparse sources    : {sparse_store.count_sources(conn)}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
