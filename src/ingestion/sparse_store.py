# src/ingestion/sparse_store.py
"""PostgreSQL-backed sparse (BM25 / full-text) index.

This is the keyword half of hybrid retrieval. Two properties matter:

1. **Stable IDs.** Rows are keyed by the *chunk's own* `Chunk.id`, the same id
   Qdrant stores for the dense vector. Reciprocal Rank Fusion deduplicates by
   id, so without a shared id the same passage retrieved both densely and
   sparsely is counted twice instead of reinforcing itself.

2. **Additive writes.** Schema creation is separated from row insertion. The
   previous implementation ran `DROP TABLE ... CASCADE` on every ingestion, so
   each upload erased the sparse index of every earlier document and BM25 only
   ever saw the newest file.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import psycopg2
from psycopg2.extras import Json, execute_batch

logger = logging.getLogger(__name__)

# Full-text search configuration used for the generated tsv column.
# 'english' applies English stemming and stopwords. Change this and the
# generated column definition together, or rebuild the table.
FTS_CONFIG = "english"

_CREATE_TABLE_SQL = f"""
    CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY,
        text TEXT NOT NULL,
        metadata JSONB,
        tsv TSVECTOR GENERATED ALWAYS AS (
            setweight(to_tsvector('{FTS_CONFIG}', coalesce(text, '')), 'A')
        ) STORED
    );
"""

_CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);"
)


def ensure_table(conn) -> None:
    """Create the chunks table and FTS index if they do not already exist.

    Idempotent and safe to call on every startup. Never destroys data.
    """
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)
        cur.execute(_CREATE_INDEX_SQL)
    conn.commit()
    logger.info("Ensured sparse (chunks) table exists in PostgreSQL")


def _row_from_chunk(chunk: Any) -> tuple[str, str, Json]:
    """Map a `Chunk` (or equivalent mapping) to a chunks-table row."""
    if isinstance(chunk, dict):
        chunk_id = chunk.get("id")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}
    else:
        chunk_id = getattr(chunk, "id", None)
        text = getattr(chunk, "text", "")
        metadata = getattr(chunk, "metadata", {}) or {}

    if chunk_id is None:
        raise ValueError("Every chunk must have an id to be stored sparsely")

    return str(chunk_id), text, Json(metadata)


def upsert_chunks(conn, chunks: Sequence[Any], batch_size: int = 200) -> int:
    """Insert or update chunks in the sparse index, keyed by chunk id.

    Uses `ON CONFLICT (id) DO UPDATE`, so re-ingesting the same chunk refreshes
    its text and metadata rather than duplicating or failing.

    Args:
        conn: An open psycopg2 connection.
        chunks: `Chunk` objects or mappings with `id`, `text`, `metadata`.
        batch_size: Rows per executemany round trip.

    Returns:
        Number of rows submitted.
    """
    chunks = list(chunks)
    if not chunks:
        # An empty write is a no-op, not an error.
        logger.info("No chunks to upsert into the sparse index")
        return 0

    rows = [_row_from_chunk(chunk) for chunk in chunks]

    sql = """
        INSERT INTO chunks (id, text, metadata)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO UPDATE
            SET text = EXCLUDED.text,
                metadata = EXCLUDED.metadata
    """

    with conn.cursor() as cur:
        execute_batch(cur, sql, rows, page_size=batch_size)
    conn.commit()

    logger.info("Upserted %d chunk(s) into the sparse index", len(rows))
    return len(rows)


def count_chunks(conn) -> int:
    """Total rows in the sparse index, for verification and diagnostics."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        return int(cur.fetchone()[0])


def count_sources(conn) -> int:
    """Number of distinct source documents represented in the sparse index."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT metadata->>'source') FROM chunks "
            "WHERE metadata->>'source' IS NOT NULL"
        )
        return int(cur.fetchone()[0])


def delete_source(conn, source: str) -> int:
    """Remove every chunk belonging to one source document.

    Useful for re-ingesting a single changed document without a full reset.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE metadata->>'source' = %s", (source,))
        deleted = cur.rowcount
    conn.commit()
    logger.info("Deleted %d chunk(s) for source %r", deleted, source)
    return deleted


def reset_index(conn) -> None:
    """DESTRUCTIVE: drop and recreate the sparse index, discarding all rows.

    Deliberately explicit and opt-in. Nothing in the ingestion path calls this,
    because silently wiping the index was the original bug.
    """
    logger.warning("Resetting the sparse index: all existing chunks are dropped")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    conn.commit()
    ensure_table(conn)


def connect(dsn: dict[str, Any] | None = None):
    """Open a connection to the sparse store using environment defaults.

    Mirrors the settings used by the API and worker so all three agree on
    which database they are talking to.
    """
    settings = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "rag_metadata"),
        "user": os.getenv("POSTGRES_USER", "raglab"),
        "password": os.getenv("POSTGRES_PASSWORD", "raglab"),
    }
    if dsn:
        settings.update(dsn)

    conn = psycopg2.connect(**settings)
    conn.autocommit = True
    return conn
