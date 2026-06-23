# tests/unit/test_hybrid_retriever.py
import pytest
from unittest.mock import MagicMock
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_dense_retriever():
    mock = MagicMock()
    # Simulate dense search results: Document A (rank 0), B (rank 1), C (rank 2)
    mock.search.return_value = [
        {"id": "A", "text": "Alpha", "score": 0.9, "metadata": {}},
        {"id": "B", "text": "Beta", "score": 0.8, "metadata": {}},
        {"id": "C", "text": "Gamma", "score": 0.7, "metadata": {}},
    ]
    return mock


@pytest.fixture
def mock_sparse_retriever():
    mock = MagicMock()
    # Simulate sparse search results: Document B (rank 0), A (rank 1), D (rank 2)
    mock.search.return_value = [
        {"id": "B", "text": "Beta", "score": 5.0, "metadata": {}},
        {"id": "A", "text": "Alpha", "score": 4.0, "metadata": {}},
        {"id": "D", "text": "Delta", "score": 3.0, "metadata": {}},
    ]
    return mock


@pytest.fixture
def config():
    return {
        "hybrid_retrieval": {
            "candidate_k": 3,
            "final_k": 2,
            "rrf_k": 60,
        }
    }


def test_hybrid_retriever_initialization(mock_dense_retriever, mock_sparse_retriever, config):
    hybrid = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        config=config,
    )
    assert hybrid.candidate_k == 3
    assert hybrid.final_k == 2
    assert hybrid.rrf_k == 60


def test_hybrid_search_calls_both_retrievers(mock_dense_retriever, mock_sparse_retriever, config):
    hybrid = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        config=config,
    )
    hybrid.search("test query")
    mock_dense_retriever.search.assert_called_once_with("test query", top_k=3)
    mock_sparse_retriever.search.assert_called_once_with("test query", top_k=3)


def test_hybrid_search_rrf_scoring(mock_dense_retriever, mock_sparse_retriever, config):
    hybrid = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        config=config,
    )
    results = hybrid.search("test query", top_k=2)

    # Expected RRF scores (k=60, ranks 0-based):
    # A: dense rank 0 -> 1/(60+0+1)=1/61=0.01639, sparse rank 1 -> 1/(60+1+1)=1/62=0.01613 => total 0.03252
    # B: dense rank 1 -> 1/62=0.01613, sparse rank 0 -> 1/61=0.01639 => total 0.03252
    # C: dense rank 2 -> 1/63=0.01587, sparse None -> 0.01587
    # D: dense None, sparse rank 2 -> 1/63=0.01587

    # A and B tie. Since we sort descending, either order is acceptable.
    # We'll check that A and B are the top 2 (order can vary, but both should be present).
    ids = [r["id"] for r in results]
    assert set(ids) == {"A", "B"}
    assert len(results) == 2
    # Check that scores are populated (floats)
    assert all("score" in r for r in results)
    assert all(isinstance(r["score"], float) for r in results)


def test_hybrid_search_empty_query(mock_dense_retriever, mock_sparse_retriever, config):
    hybrid = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        config=config,
    )
    results = hybrid.search("")
    assert results == []
    mock_dense_retriever.search.assert_not_called()
    mock_sparse_retriever.search.assert_not_called()


def test_hybrid_search_handles_missing_results(mock_dense_retriever, mock_sparse_retriever, config):
    # Simulate sparse retriever returning empty list
    mock_sparse_retriever.search.return_value = []
    hybrid = HybridRetriever(
        dense_retriever=mock_dense_retriever,
        sparse_retriever=mock_sparse_retriever,
        config=config,
    )
    results = hybrid.search("test query", top_k=2)
    # Should still return dense results (fused with nothing)
    assert len(results) == 2
    assert results[0]["id"] == "A"
    assert results[1]["id"] == "B"


def test_hybrid_search_respects_final_k(config):
    # Override config for this test
    config["hybrid_retrieval"]["final_k"] = 1
    mock_dense = MagicMock()
    mock_dense.search.return_value = [
        {"id": "A", "text": "Alpha", "score": 0.9, "metadata": {}},
        {"id": "B", "text": "Beta", "score": 0.8, "metadata": {}},
    ]
    mock_sparse = MagicMock()
    mock_sparse.search.return_value = [
        {"id": "B", "text": "Beta", "score": 5.0, "metadata": {}},
    ]
    hybrid = HybridRetriever(
        dense_retriever=mock_dense,
        sparse_retriever=mock_sparse,
        config=config,
    )
    results = hybrid.search("test query")
    assert len(results) == 1