# tests/unit/test_multi_query_retriever.py
import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    # Simulate LLM response: 3 variations
    mock.invoke.return_value = MagicMock(
        content="What is the launch date of JWST?\nWhen was JWST launched?\nJWST launch year and date?"
    )
    return mock


@pytest.fixture
def mock_base_retriever():
    mock = MagicMock(spec=HybridRetriever)
    # Simulate search results for each query variant
    def search_side_effect(query, top_k):
        # For simplicity, return different mock results based on the query
        if "launch date" in query.lower():
            return [
                {"id": "doc_1", "text": "JWST launched on Dec 25, 2021", "score": 0.95, "metadata": {}},
                {"id": "doc_2", "text": "JWST is at L2", "score": 0.80, "metadata": {}},
            ]
        elif "when" in query.lower():
            return [
                {"id": "doc_1", "text": "JWST launched on Dec 25, 2021", "score": 0.90, "metadata": {}},
                {"id": "doc_3", "text": "JWST was launched in 2021", "score": 0.85, "metadata": {}},
            ]
        else:
            return [
                {"id": "doc_2", "text": "JWST is at L2", "score": 0.75, "metadata": {}},
                {"id": "doc_4", "text": "JWST is a space telescope", "score": 0.70, "metadata": {}},
            ]
    mock.search.side_effect = search_side_effect
    return mock


@pytest.fixture
def multi_query_retriever(mock_llm, mock_base_retriever):
    config = {
        "multi_query": {
            "enabled": True,
            "num_queries": 3,
            "candidate_k": 10,
            "temperature": 0.0,
            "rrf_k": 60,
        }
    }
    return MultiQueryRetriever(
        llm=mock_llm,
        base_retriever=mock_base_retriever,
        config=config,
    )


def test_multi_query_initialization(multi_query_retriever, mock_llm, mock_base_retriever):
    assert multi_query_retriever.llm == mock_llm
    assert multi_query_retriever.base_retriever == mock_base_retriever
    assert multi_query_retriever.num_queries == 3
    assert multi_query_retriever.candidate_k == 10


def test_generate_queries(multi_query_retriever, mock_llm):
    queries = multi_query_retriever._generate_queries("JWST launch date?")
    mock_llm.invoke.assert_called_once()
    assert len(queries) == 3
    assert queries[0] == "What is the launch date of JWST?"
    assert queries[1] == "When was JWST launched?"
    assert queries[2] == "JWST launch year and date?"


def test_search_calls_base_retriever_for_each_variant(multi_query_retriever, mock_base_retriever):
    results = multi_query_retriever.search("JWST", top_k=3)
    # Base retriever should be called 3 times (for each variant)
    assert mock_base_retriever.search.call_count == 3
    # Verify results are returned
    assert len(results) == 3  # top_k=3


def test_search_fusion_works(multi_query_retriever):
    # We'll test the fusion logic indirectly: doc_1 should appear in multiple sets, so it should rank high.
    results = multi_query_retriever.search("JWST", top_k=2)
    # doc_1 appears in the first two result sets, so it should be the top result.
    assert results[0]["id"] == "doc_1"
    # doc_2 appears in set 1 and 3, so it might be second.
    # The exact order can vary, but we check that doc_1 is first.
    assert results[0]["score"] > 0


def test_multi_query_disabled(multi_query_retriever, mock_base_retriever):
    # Override config to disable
    config = {"multi_query": {"enabled": False}}
    retriever = MultiQueryRetriever(
        llm=multi_query_retriever.llm,
        base_retriever=mock_base_retriever,
        config=config,
    )
    retriever.search("JWST", top_k=3)
    # Base retriever should be called only once, with the original query
    mock_base_retriever.search.assert_called_once_with("JWST", top_k=3)
    # LLM should NOT be called
    multi_query_retriever.llm.invoke.assert_not_called()


def test_search_empty_query(multi_query_retriever):
    results = multi_query_retriever.search("")
    assert results == []
    multi_query_retriever.llm.invoke.assert_not_called()


def test_search_respects_top_k(multi_query_retriever):
    results = multi_query_retriever.search("JWST", top_k=1)
    assert len(results) == 1