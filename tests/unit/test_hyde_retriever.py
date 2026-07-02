import pytest
from unittest.mock import MagicMock
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.hybrid_retriever import HybridRetriever


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="The James Webb Space Telescope launched on December 25, 2021."
    )
    return mock


@pytest.fixture
def mock_embedder():
    # IMPORTANT: should NOT be used by HyDE layer in your architecture
    return MagicMock()


@pytest.fixture
def mock_base_retriever():
    mock = MagicMock(spec=HybridRetriever)
    mock.search.return_value = [
        {"id": "doc_1", "text": "JWST launched on Dec 25, 2021", "score": 0.95},
        {"id": "doc_2", "text": "JWST is at L2", "score": 0.85},
    ]
    return mock


@pytest.fixture
def hyde_retriever(mock_llm, mock_embedder, mock_base_retriever):
    config = {
        "hyde": {
            "enabled": True,
            "temperature": 0.0,
            "prompt_template": "Question: {query}",
        }
    }

    return HyDERetriever(
        llm=mock_llm,
        embedder=mock_embedder,
        base_retriever=mock_base_retriever,
        config=config,
    )


# -----------------------
# Tests
# -----------------------

def test_hyde_generates_hypothetical_doc(hyde_retriever, mock_llm):
    result = hyde_retriever._generate_hypothetical("JWST launch date")

    mock_llm.invoke.assert_called_once()

    assert result == "The James Webb Space Telescope launched on December 25, 2021."


def test_search_uses_hyde_rewritten_query(hyde_retriever, mock_base_retriever):
    hyde_retriever.search("JWST launch date", top_k=3)

    mock_base_retriever.search.assert_called_once()

    args, kwargs = mock_base_retriever.search.call_args

    # HyDE rewrites query → passed to retriever
    assert args[0] == "The James Webb Space Telescope launched on December 25, 2021."
    assert kwargs["top_k"] == 3


def test_embedder_not_used_in_hyde_layer(hyde_retriever, mock_embedder):
    hyde_retriever.search("JWST launch date", top_k=3)

    # IMPORTANT: HyDE does NOT do embedding in your architecture
    mock_embedder.embed.assert_not_called()


def test_search_returns_base_results(hyde_retriever, mock_base_retriever):
    results = hyde_retriever.search("JWST launch date", top_k=2)

    assert results == mock_base_retriever.search.return_value


def test_hyde_disabled_bypasses_llm(mock_llm, mock_embedder, mock_base_retriever):
    config = {"hyde": {"enabled": False}}

    retriever = HyDERetriever(
        llm=mock_llm,
        embedder=mock_embedder,
        base_retriever=mock_base_retriever,
        config=config,
    )

    retriever.search("JWST launch date", top_k=3)

    # LLM must NOT be used
    mock_llm.invoke.assert_not_called()

    # Must pass original query directly
    mock_base_retriever.search.assert_called_once_with(
        "JWST launch date",
        top_k=3,
    )