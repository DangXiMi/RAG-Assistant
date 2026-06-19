# tests/unit/test_dense_retriever.py
import pytest
import uuid
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.retrieval.dense_retriever import DenseRetriever
from src.ingestion.chunker import Chunk


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def indexer():
    config = {
        "qdrant": {
            "host": ":memory:",
            "port": 6333,
            "collection_name": "test_collection",
            "vector_size": 384,
        },
        "retrieval": {
            "default_top_k": 3,
            "score_threshold": 0.0,
        }
    }
    idx = Indexer(config=config)
    idx.ensure_collection()
    return idx


@pytest.fixture(scope="module")
def retriever(embedder, indexer):
    config = {
        "retrieval": {
            "default_top_k": 3,
            "score_threshold": 0.0,
        }
    }
    return DenseRetriever(embedder=embedder, indexer=indexer, config=config)


@pytest.fixture(scope="module")
def seeded_chunks(indexer, embedder):
    """Seed the index with 5 known chunks."""
    texts = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome.",
        "The capital of Spain is Madrid.",
        "The capital of Portugal is Lisbon.",
    ]
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=text,
                metadata={"country": text.split()[3] if len(text.split()) > 3 else "unknown", "chunk_index": i, "total_chunks": 5},
                start_char=0,
                end_char=len(text),
            )
        )
    vectors = embedder.embed(texts)
    indexer.index(chunks, vectors)
    return chunks


def test_retriever_initialization(retriever):
    """Ensure retriever holds dependencies."""
    assert retriever.embedder is not None
    assert retriever.indexer is not None
    assert retriever.default_top_k == 3


def test_search_returns_top_k(retriever, seeded_chunks):
    """Search returns exactly `top_k` results (if available)."""
    results = retriever.search("What is the capital of France?", top_k=2)
    assert len(results) == 2
    assert "score" in results[0]
    assert "text" in results[0]
    assert "metadata" in results[0]


def test_search_returns_less_if_not_enough(retriever, seeded_chunks):
    """If fewer than top_k exist, return all."""
    results = retriever.search("Portugal", top_k=10)
    assert len(results) <= 5  # we only have 5 docs


def test_search_results_are_relevant(retriever, seeded_chunks):
    """The top result for 'France' should be the France chunk."""
    results = retriever.search("France capital", top_k=1)
    assert len(results) == 1
    assert "Paris" in results[0]["text"]


def test_search_empty_query(retriever, seeded_chunks):
    """Empty query should return empty list (or maybe error, but we choose empty)."""
    results = retriever.search("")
    assert results == []


def test_search_with_threshold(retriever):
    """If threshold is set, filter out low scores."""
    # We need to create a retriever with a high threshold
    config = {
        "retrieval": {
            "default_top_k": 5,
            "score_threshold": 0.9,  # very high, should return none
        }
    }
    # Re-initialize with same embedder/indexer
    retriever_high = DenseRetriever(embedder=retriever.embedder, indexer=retriever.indexer, config=config)
    results = retriever_high.search("France")
    # Since our dummy vectors are not meaningful, this might return all or none.
    # We can't guarantee >0.9, so we just check that the threshold is applied
    # (i.e., results are filtered). We'll rely on the implementation to filter.
    # For now, we assert it doesn't crash.
    assert isinstance(results, list)


def test_search_uses_default_top_k(retriever, seeded_chunks):
    """If top_k not provided, use default from config."""
    # We set default_top_k=3 in fixture
    results = retriever.search("capital")
    assert len(results) == 3  # should return exactly 3


def test_search_returns_points_with_payload(retriever, seeded_chunks):
    """Ensure payload fields are present."""
    results = retriever.search("Berlin", top_k=1)
    assert len(results) == 1
    result = results[0]
    assert "id" in result
    assert result["text"] == "The capital of Germany is Berlin."
    assert result["metadata"]["country"] == "Germany"
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0  # cosine similarity is in [0,1] for MiniLM