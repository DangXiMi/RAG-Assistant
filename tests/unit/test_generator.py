# tests/unit/test_generator.py
import pytest
from unittest.mock import MagicMock, patch
from src.generation.generator import Generator
from langchain_core.documents import Document


@pytest.fixture
def mock_retriever():
    mock = MagicMock()
    # Simulate hybrid retriever results (format returned by your HybridRetriever)
    mock.search.return_value = [
        {"id": "doc_1", "text": "Paris is the capital of France.", "score": 0.95, "metadata": {"source": "wiki"}},
        {"id": "doc_2", "text": "Berlin is the capital of Germany.", "score": 0.85, "metadata": {"source": "wiki"}},
    ]
    return mock


@pytest.fixture
def config():
    return {
        "generator": {  # Note: "generator" not "generation" – matches your code
            "model": "llama3.1:8b",
            "temperature": 0.0,
            "max_tokens": 512,
        }
    }


@pytest.fixture
def generator(mock_retriever, config):
    # We'll mock ChatOllama inside each test. For the fixture, we just instantiate.
    # But we need to patch ChatOllama globally. We'll do it per test via patch.
    return Generator(retriever=mock_retriever, config=config)


def test_generator_initialization(generator, mock_retriever, config):
    assert generator.retriever == mock_retriever
    assert generator.model_name == config["generator"]["model"]
    assert generator.temperature == config["generator"]["temperature"]
    assert generator.max_tokens == config["generator"]["max_tokens"]
    assert generator.model is not None  # ChatOllama instance


def test_retrieve_converts_to_langchain_documents(generator, mock_retriever):
    docs = generator.retrieve("test query", top_k=2)
    
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Paris is the capital of France."
    assert docs[0].metadata["doc_id"] == "doc_1"
    assert docs[0].metadata["source"] == "wiki"
    
    # Verify retriever was called with correct params
    mock_retriever.search.assert_called_once_with(query="test query", top_k=2)


def test_build_context_formats_correctly(generator):
    docs = [
        Document(page_content="Content A", metadata={"doc_id": "id_a"}),
        Document(page_content="Content B", metadata={"doc_id": "id_b"}),
    ]
    context = generator.build_context(docs)
    
    expected = "[id_a]\nContent A\n\n[id_b]\nContent B"
    assert context == expected


@patch("src.generation.generator.ChatOllama")
def test_generate_calls_model_invoke(mock_ollama_class, generator):
    # Mock the model instance
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Paris is the capital."
    mock_model_instance.invoke.return_value = mock_response
    mock_ollama_class.return_value = mock_model_instance

    # Override the model in the generator instance (since it was created in __init__)
    generator.model = mock_model_instance

    context = "[doc_1]\nParis is the capital of France."
    response = generator.generate(context, "What is the capital of France?")

    # Verify invoke was called
    mock_model_instance.invoke.assert_called_once()
    
    # Check that the response content is correctly returned
    # Note: generator.generate returns the BaseMessage (or similar), not a string.
    # We'll test that the invoke happened and the content is correct.
    assert response.content == "Paris is the capital."


@patch("src.generation.generator.ChatOllama")
def test_run_orchestrates_pipeline(mock_ollama_class, generator, mock_retriever):
    # Mock the model
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Paris is the capital."
    mock_model_instance.invoke.return_value = mock_response
    mock_ollama_class.return_value = mock_model_instance
    generator.model = mock_model_instance

    result = generator.run("What is the capital of France?", top_k=2)

    # Verify retriever was called
    mock_retriever.search.assert_called_once_with(query="What is the capital of France?", top_k=2)
    
    # Verify model was invoked
    mock_model_instance.invoke.assert_called_once()
    
    # Check result structure
    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "Paris is the capital."
    assert result["sources"] == ["doc_1", "doc_2"]


@patch("src.generation.generator.ChatOllama")
def test_run_handles_empty_retrieval(mock_ollama_class, generator, mock_retriever):
    # Simulate empty retrieval
    mock_retriever.search.return_value = []
    
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "I don't know."
    mock_model_instance.invoke.return_value = mock_response
    mock_ollama_class.return_value = mock_model_instance
    generator.model = mock_model_instance

    result = generator.run("What is the capital of nowhere?", top_k=2)

    # The context should be empty (just ""), and the LLM should be called with empty context
    mock_model_instance.invoke.assert_called_once()
    assert result["answer"] == "I don't know."
    assert result["sources"] == []


@patch("src.generation.generator.ChatOllama")
def test_run_uses_default_top_k_if_not_provided(mock_ollama_class, generator, mock_retriever):
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_model_instance.invoke.return_value = mock_response
    mock_ollama_class.return_value = mock_model_instance
    generator.model = mock_model_instance

    # top_k defaults to 5 in your run method
    generator.run("test query")
    mock_retriever.search.assert_called_once_with(query="test query", top_k=5)


def test_build_context_with_empty_docs(generator):
    context = generator.build_context([])
    assert context == ""