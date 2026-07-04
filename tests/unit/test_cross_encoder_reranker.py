# tests/unit/test_cross_encoder_reranker.py
import pytest
from unittest.mock import MagicMock, patch
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_base_retriever():
    mock = MagicMock(spec=HybridRetriever)
    # Simulate base retriever returning 3 candidates
    mock.search.return_value = [
        {"id": "doc_1", "text": "The capital of France is Paris.", "score": 0.85, "metadata": {}},
        {"id": "doc_2", "text": "The capital of Germany is Berlin.", "score": 0.80, "metadata": {}},
        {"id": "doc_3", "text": "The capital of Italy is Rome.", "score": 0.75, "metadata": {}},
    ]
    return mock


@pytest.fixture
def reranker(mock_base_retriever):
    config = {
        "reranking": {
            "enabled": True,
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidate_k": 3,
            "final_k": 2,
            "batch_size": 32,
        }
    }
    # We'll mock the CrossEncoder inside the tests to avoid loading the model
    return CrossEncoderReranker(base_retriever=mock_base_retriever, config=config)


@patch("src.reranking.cross_encoder_reranker.CrossEncoder")
def test_reranker_initialization(mock_cross_encoder, mock_base_retriever, reranker):
    # The __init__ will call CrossEncoder with the model name from config
    # We need to check that it was called
    #mock_cross_encoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker.base_retriever == mock_base_retriever
    assert reranker.candidate_k == 3
    assert reranker.final_k == 2
    assert reranker.batch_size == 32


@patch("src.reranking.cross_encoder_reranker.CrossEncoder")
def test_rerank_sorts_by_cross_encoder_score(mock_cross_encoder, reranker):
    # Mock the cross-encoder's predict method to return scores
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.1, 0.5]  # scores for the 3 candidates
    mock_cross_encoder.return_value = mock_model
    reranker.model = mock_model  # Override the model loaded in __init__

    candidates = [
        {"id": "doc_1", "text": "Paris", "metadata": {}},
        {"id": "doc_2", "text": "Berlin", "metadata": {}},
        {"id": "doc_3", "text": "Rome", "metadata": {}},
    ]
    reranked = reranker.rerank("France", candidates, top_k=2)
    # Should return top 2 based on scores (doc_1: 0.9, doc_3: 0.5)
    assert len(reranked) == 2
    assert reranked[0]["id"] == "doc_1"
    assert reranked[1]["id"] == "doc_3"
    # Check that scores are updated
    assert reranked[0]["score"] == 0.9
    assert reranked[1]["score"] == 0.5


@patch("src.reranking.cross_encoder_reranker.CrossEncoder")
def test_search_retrieves_candidates_and_reranks(mock_cross_encoder, reranker, mock_base_retriever):
    # Mock the cross-encoder scores
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.1, 0.5]
    mock_cross_encoder.return_value = mock_model
    reranker.model = mock_model

    results = reranker.search("France", top_k=1)
    # Base retriever should be called with candidate_k (3)
    mock_base_retriever.search.assert_called_once_with("France", top_k=3)
    # Final results should be top 1
    assert len(results) == 1
    assert results[0]["id"] == "doc_1"
    assert results[0]["score"] == 0.9


@patch("src.reranking.cross_encoder_reranker.CrossEncoder")
def test_search_empty_query(mock_cross_encoder, reranker):
    results = reranker.search("")
    assert results == []
    reranker.base_retriever.search.assert_not_called()


@patch("src.reranking.cross_encoder_reranker.CrossEncoder")
def test_search_respects_final_k(mock_cross_encoder, reranker):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
    mock_cross_encoder.return_value = mock_model
    reranker.model = mock_model

    # Configure final_k=2 via config (already set in fixture)
    results = reranker.search("France", top_k=None)  # should use final_k=2
    assert len(results) == 2


@patch("src.reranking.cross_encoder_reranker.CrossEncoder")
def test_reranker_disabled(mock_cross_encoder, reranker):
    # Override config to disable
    config = {"reranking": {"enabled": False}}
    disabled_reranker = CrossEncoderReranker(
        base_retriever=reranker.base_retriever,
        config=config,
    )
    # When disabled, search should just call base_retriever and return its results without reranking
    results = disabled_reranker.search("France", top_k=2)
    disabled_reranker.base_retriever.search.assert_called_once_with("France", top_k=2)
    # No cross-encoder should be loaded
    assert disabled_reranker.model is None
    # Results should be exactly the base retriever results
    assert len(results) == 3  # base returns 3, and we didn't rerank