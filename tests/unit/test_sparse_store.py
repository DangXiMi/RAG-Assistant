# tests/unit/test_sparse_store.py
"""Tests for the Postgres sparse (BM25) index.

Tests that need a live database are skipped when PostgreSQL is unreachable, so
the suite stays green on a machine without the backing services.
"""
import os
import uuid

import pytest

from src.ingestion.chunker import Chunk
from src.ingestion import sparse_store


def _postgres_available() -> bool:
    try:
        conn = sparse_store.connect()
        conn.close()
        return True
    except Exception:
        return False


POSTGRES = _postgres_available()
requires_postgres = pytest.mark.skipif(
    not POSTGRES,
    reason="PostgreSQL is not reachable; start it to run sparse-store tests",
)


def make_chunk(text, source="a.pdf", page=1, chunk_id=None, **extra):
    metadata = {"source": source, "page": page, **extra}
    return Chunk(
        id=chunk_id or str(uuid.uuid4()),
        text=text,
        metadata=metadata,
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def conn():
    connection = sparse_store.connect()
    yield connection
    connection.close()


# --- Pure logic (no database) ----------------------------------------------

def test_row_from_chunk_reads_attributes():
    chunk = make_chunk("hello world", source="x.pdf", page=2)

    chunk_id, text, metadata = sparse_store._row_from_chunk(chunk)

    assert chunk_id == str(chunk.id)
    assert text == "hello world"
    assert metadata.adapted["source"] == "x.pdf"


def test_row_from_chunk_accepts_a_mapping():
    payload = {"id": "abc", "text": "t", "metadata": {"source": "m.pdf"}}

    chunk_id, text, _ = sparse_store._row_from_chunk(payload)

    assert chunk_id == "abc"
    assert text == "t"


def test_row_from_chunk_rejects_a_missing_id():
    with pytest.raises(ValueError):
        sparse_store._row_from_chunk({"text": "no id", "metadata": {}})


def test_fts_config_is_declared():
    assert sparse_store.FTS_CONFIG == "english"
    assert "to_tsvector" in sparse_store._CREATE_TABLE_SQL


def test_create_sql_is_not_destructive():
    """The original bug was a DROP inside the ingest path."""
    assert "DROP" not in sparse_store._CREATE_TABLE_SQL.upper()
    assert "DROP" not in sparse_store._CREATE_INDEX_SQL.upper()


# --- Database-backed behaviour ---------------------------------------------

@requires_postgres
def test_ensure_table_is_idempotent(conn):
    sparse_store.ensure_table(conn)
    sparse_store.ensure_table(conn)

    assert sparse_store.count_chunks(conn) >= 0


@requires_postgres
def test_upsert_then_count(conn):
    sparse_store.ensure_table(conn)
    source = f"test-{uuid.uuid4()}.txt"
    chunks = [make_chunk(f"unique text one {uuid.uuid4()}", source=source, page=1),
              make_chunk(f"unique text two {uuid.uuid4()}", source=source, page=2)]

    inserted = sparse_store.upsert_chunks(conn, chunks)

    assert inserted == 2
    sparse_store.delete_source(conn, source)


@requires_postgres
def test_upsert_stores_the_real_chunk_id(conn):
    """Item 7: the sparse id must equal the chunk id, so RRF can fuse."""
    sparse_store.ensure_table(conn)
    chunk_id = str(uuid.uuid4())
    source = f"test-{uuid.uuid4()}.txt"
    chunk = make_chunk("identity check", source=source, chunk_id=chunk_id)

    sparse_store.upsert_chunks(conn, [chunk])

    with conn.cursor() as cur:
        cur.execute("SELECT id::text FROM chunks WHERE id = %s", (chunk_id,))
        row = cur.fetchone()

    assert row is not None, "chunk id was not used as the primary key"
    assert row[0] == chunk_id
    sparse_store.delete_source(conn, source)


@requires_postgres
def test_upsert_is_idempotent_and_updates(conn):
    sparse_store.ensure_table(conn)
    chunk_id = str(uuid.uuid4())
    source = f"test-{uuid.uuid4()}.txt"

    sparse_store.upsert_chunks(conn, [make_chunk("original text", source=source, chunk_id=chunk_id)])
    sparse_store.upsert_chunks(conn, [make_chunk("updated text", source=source, chunk_id=chunk_id)])

    with conn.cursor() as cur:
        cur.execute("SELECT count(*), max(text) FROM chunks WHERE id = %s", (chunk_id,))
        count, text = cur.fetchone()

    assert count == 1, "re-ingesting the same chunk duplicated the row"
    assert text == "updated text"
    sparse_store.delete_source(conn, source)


@requires_postgres
def test_upsert_preserves_location_metadata(conn):
    sparse_store.ensure_table(conn)
    source = f"test-{uuid.uuid4()}.xlsx"
    chunk = make_chunk("sheet data", source=source, sheet="Costs", row="1-4")

    sparse_store.upsert_chunks(conn, [chunk])

    with conn.cursor() as cur:
        cur.execute(
            "SELECT metadata->>'sheet', metadata->>'row' FROM chunks WHERE id = %s",
            (str(chunk.id),),
        )
        sheet, row = cur.fetchone()

    assert sheet == "Costs"
    assert row == "1-4"
    sparse_store.delete_source(conn, source)


@requires_postgres
def test_upsert_of_nothing_is_a_noop(conn):
    sparse_store.ensure_table(conn)

    assert sparse_store.upsert_chunks(conn, []) == 0


@requires_postgres
def test_ingesting_a_second_document_does_not_erase_the_first(conn):
    """Item 8, the core regression.

    Previously every ingest ran DROP TABLE, so the sparse index only ever held
    the newest file.
    """
    sparse_store.ensure_table(conn)
    source_a = f"docA-{uuid.uuid4()}.txt"
    source_b = f"docB-{uuid.uuid4()}.txt"

    sparse_store.upsert_chunks(conn, [make_chunk("alpha content about refunds", source=source_a)])
    sparse_store.upsert_chunks(conn, [make_chunk("beta content about shipping", source=source_b)])

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT metadata->>'source') FROM chunks "
            "WHERE metadata->>'source' IN (%s, %s)",
            (source_a, source_b),
        )
        distinct = cur.fetchone()[0]

    assert distinct == 2, "ingesting B removed A from the sparse index"

    sparse_store.delete_source(conn, source_a)
    sparse_store.delete_source(conn, source_b)


@requires_postgres
def test_delete_source_removes_only_that_document(conn):
    sparse_store.ensure_table(conn)
    source_a = f"delA-{uuid.uuid4()}.txt"
    source_b = f"delB-{uuid.uuid4()}.txt"

    sparse_store.upsert_chunks(conn, [make_chunk("text a", source=source_a)])
    sparse_store.upsert_chunks(conn, [make_chunk("text b", source=source_b)])

    deleted = sparse_store.delete_source(conn, source_a)

    assert deleted == 1
    assert sparse_store.count_sources(conn) >= 1
    sparse_store.delete_source(conn, source_b)


@requires_postgres
def test_count_sources_reports_distinct_documents(conn):
    sparse_store.ensure_table(conn)
    source = f"count-{uuid.uuid4()}.txt"

    sparse_store.upsert_chunks(conn, [
        make_chunk(f"one {uuid.uuid4()}", source=source, page=1),
        make_chunk(f"two {uuid.uuid4()}", source=source, page=2),
    ])

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT metadata->>'source') FROM chunks "
            "WHERE metadata->>'source' = %s",
            (source,),
        )
        assert cur.fetchone()[0] == 1

    sparse_store.delete_source(conn, source)
