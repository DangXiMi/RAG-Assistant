# tests/integration/test_sparse_retriever.py
import os
import uuid
import pytest
import psycopg2
from psycopg2.extras import Json
from src.retrieval.sparse_retriever import SparseRetriever


@pytest.fixture(scope="session")
def db_conn():
    """Connect to test Postgres database."""
    # Use environment variables or fallback to defaults
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag_metadata"),
        user=os.getenv("POSTGRES_USER", "raglab"),
        password=os.getenv("POSTGRES_PASSWORD", "raglab")
    )
    conn.autocommit = True  # for test setup
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def setup_table(db_conn):
    """Create the chunks table with FTS and insert sample data."""
    cur = db_conn.cursor()
    # Drop table if exists (clean slate)
    cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    # Create table with generated tsvector column
    cur.execute("""
        CREATE TABLE chunks (
            id UUID PRIMARY KEY,
            text TEXT NOT NULL,
            metadata JSONB,
            tsv TSVECTOR GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(text, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(metadata->>'country', '')), 'B')
            ) STORED
        );
    """)
    # Create GIN index for fast search
    cur.execute("CREATE INDEX idx_chunks_tsv ON chunks USING GIN (tsv);")

    # Insert sample documents
    samples = [
        ("The capital of France is Paris.", {"country": "France"}),
        ("The capital of Germany is Berlin.", {"country": "Germany"}),
        ("The capital of Italy is Rome.", {"country": "Italy"}),
        ("The capital of Spain is Madrid.", {"country": "Spain"}),
        ("The capital of Portugal is Lisbon.", {"country": "Portugal"}),
    ]
    for text, meta in samples:
        cur.execute(
            "INSERT INTO chunks (id, text, metadata) VALUES (%s, %s, %s)",
            (str(uuid.uuid4()), text, Json(meta))
        )
    db_conn.commit()
    cur.close()
    yield
    # Teardown: drop table
    cur = db_conn.cursor()
    cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    db_conn.commit()
    cur.close()


@pytest.fixture
def retriever(db_conn):
    config = {
        "sparse_retrieval": {
            "default_top_k": 3,
            "score_threshold": 0.0,
        }
    }
    return SparseRetriever(db_conn=db_conn, config=config)


def test_retriever_initialization(retriever):
    assert retriever.default_top_k == 3
    assert retriever.score_threshold == 0.0


def test_search_returns_top_k(retriever, setup_table):
    results = retriever.search("France capital", top_k=2)
    assert len(results) == 1
    assert "score" in results[0]
    assert "text" in results[0]
    assert "metadata" in results[0]


def test_search_returns_less_if_not_enough(retriever, setup_table):
    results = retriever.search("Portugal", top_k=10)
    assert len(results) == 1  # only one Portugal doc


def test_search_results_are_relevant(retriever, setup_table):
    results = retriever.search("France", top_k=1)
    assert len(results) == 1
    assert "Paris" in results[0]["text"]


def test_search_empty_query(retriever, setup_table):
    results = retriever.search("")
    assert results == []


def test_search_with_threshold(retriever, setup_table):
    # Set threshold high to filter out all results
    config = {
        "sparse_retrieval": {
            "default_top_k": 5,
            "score_threshold": 10.0,  # ts_rank usually < 1 for these small docs
        }
    }
    retriever_high = SparseRetriever(db_conn=retriever.db_conn, config=config)
    results = retriever_high.search("France")
    assert len(results) == 0


def test_search_uses_default_top_k(retriever, setup_table):
    results = retriever.search("capital")
    assert len(results) == 3  # default_top_k = 3


def test_search_returns_points_with_payload(retriever, setup_table):
    results = retriever.search("Berlin", top_k=1)
    assert len(results) == 1
    result = results[0]
    assert "id" in result
    assert result["text"] == "The capital of Germany is Berlin."
    assert result["metadata"]["country"] == "Germany"
    assert "score" in result
    assert result["score"] > 0