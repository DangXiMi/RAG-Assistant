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
    # `retriever` is passed by keyword here for the same reason `run` does it:
    # the @traceable wrapper resolves arguments by name.
    docs = generator.retrieve("test query", retriever=mock_retriever, top_k=2)
    
    assert len(docs) == 2
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Paris is the capital of France."
    assert docs[0].metadata["doc_id"] == "doc_1"
    assert docs[0].metadata["source"] == "wiki"
    
    # Verify retriever was called with correct params
    mock_retriever.search.assert_called_once_with(query="test query", top_k=2)


def test_retrieve_preserves_location_metadata(generator):
    """Item 4: `source` and `page` must survive retrieval to be citable."""
    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "id": "chunk-1",
            "text": "Refunds are processed within 30 days.",
            "score": 0.9,
            "metadata": {"source": "refund_policy.pdf", "page": 3, "sheet": None},
        }
    ]

    docs = generator.retrieve("refund window", retriever, top_k=1)

    assert docs[0].metadata["source"] == "refund_policy.pdf"
    assert docs[0].metadata["page"] == 3
    assert docs[0].metadata["score"] == 0.9
    # The id is exposed under both established names.
    assert docs[0].metadata["doc_id"] == "chunk-1"
    assert docs[0].metadata["id"] == "chunk-1"


def test_build_context_formats_correctly(generator):
    docs = [
        Document(
            page_content="Content A",
            metadata={"doc_id": "id_a", "source": "a.pdf", "page": 3},
        ),
        Document(
            page_content="Content B",
            metadata={"doc_id": "id_b", "source": "b.xlsx", "sheet": "Costs"},
        ),
    ]
    context = generator.build_context(docs)

    expected = (
        "[1] a.pdf, p.3\nContent A\n\n"
        "[2] b.xlsx, sheet 'Costs'\nContent B"
    )
    assert context == expected


def test_build_context_uses_readable_references_not_uuids(generator):
    """The regression this change fixes: context used to be labelled by UUID."""
    docs = [
        Document(
            page_content="Policy text",
            metadata={"doc_id": "9f8c1e2a-uuid", "source": "policy.pdf", "page": 1},
        )
    ]

    context = generator.build_context(docs)

    assert "policy.pdf, p.1" in context
    assert "9f8c1e2a-uuid" not in context


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
    # Both fixture chunks come from the same source with no page/sheet, so they
    # collapse to a single citation: repeating "wiki" twice tells a reader
    # nothing. Deduplication is deliberate.
    assert result["sources"] == ["[1] wiki"]
    # Structured detail is additive alongside the plain-string sources.
    assert result["source_refs"][0]["source"] == "wiki"
    assert result["source_refs"][0]["chunk_id"] == "doc_1"
    assert result["answered"] is True
    assert result["retrieved"] == 2


@patch("src.generation.generator.ChatOllama")
def test_run_handles_empty_retrieval(mock_ollama_class, generator, mock_retriever):
    """Item 12: with no evidence, answer deterministically without the LLM."""
    mock_retriever.search.return_value = []

    mock_model_instance = MagicMock()
    mock_model_instance.invoke.return_value = MagicMock(content="should not be used")
    generator.model = mock_model_instance

    result = generator.run("What is the capital of nowhere?", top_k=2)

    # The LLM must NOT be consulted when there is nothing to ground on.
    mock_model_instance.invoke.assert_not_called()
    assert result["answered"] is False
    assert result["retrieved"] == 0
    assert "could not find" in result["answer"].lower()
    assert result["sources"] == []
    assert result["contexts"] == []
    assert result["source_refs"] == []


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


# --- Score threshold filtering (item 12) ------------------------------------

def test_no_documents_returns_not_found_without_llm(mock_retriever):
    """A threshold-disabled generator still short-circuits on empty retrieval."""
    mock_retriever.search.return_value = []
    config = {"generator": {"model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 64}}
    gen = Generator(retriever=mock_retriever, config=config)
    gen.model = MagicMock()

    result = gen.run("anything", top_k=3)

    gen.model.invoke.assert_not_called()
    assert result["answered"] is False


def test_low_scores_are_filtered_out(mock_retriever):
    """Hits below retrieval.score_threshold must not reach the LLM."""
    config = {
        "generator": {"model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 64},
        "retrieval": {"score_threshold": 0.5},
    }
    mock_retriever.search.return_value = [
        {"id": "weak", "text": "barely related", "score": 0.01,
         "metadata": {"source": "a.pdf", "page": 1}},
    ]
    gen = Generator(retriever=mock_retriever, config=config)
    gen.model = MagicMock()

    result = gen.run("question", top_k=3)

    gen.model.invoke.assert_not_called()
    assert result["answered"] is False
    assert result["retrieved"] == 0


def test_scores_at_or_above_threshold_pass(mock_retriever):
    config = {
        "generator": {"model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 64},
        "retrieval": {"score_threshold": 0.5},
    }
    mock_retriever.search.return_value = [
        {"id": "strong", "text": "refund window is 30 days", "score": 0.9,
         "metadata": {"source": "policy.pdf", "page": 2}},
    ]
    gen = Generator(retriever=mock_retriever, config=config)
    gen.model = MagicMock()
    gen.model.invoke.return_value = MagicMock(content="30 days")

    result = gen.run("refund window?", top_k=3)

    gen.model.invoke.assert_called_once()
    assert result["answered"] is True
    assert result["sources"] == ["[1] policy.pdf, p.2"]


def test_filter_by_score_keeps_docs_without_a_score():
    """A missing score must not silently discard otherwise valid evidence."""
    config = {
        "generator": {"model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 64},
        "retrieval": {"score_threshold": 0.5},
    }
    retriever = MagicMock()
    gen = Generator(retriever=retriever, config=config)

    kept = gen.filter_by_score(
        [Document(page_content="x", metadata={"source": "a.pdf"})]
    )

    assert len(kept) == 1


# --- Answer-quality instruction (measured improvement) ----------------------

def test_prompt_asks_for_specific_facts_first(mock_retriever):
    """The prompt must push for concrete facts over generalities.

    Regression guard for a measured fix. On the assignment's example question
    "What is the refund policy?", the old prompt produced only
    "We want you to be completely satisfied with your purchase." and omitted
    the 30-day window that was present in the retrieved context. Adding this
    instruction took the demo question set from 7/8 to 8/8 correct answers,
    verified by scripts/measure_answers.py before and after.
    """
    config = {"generator": {"model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 64}}
    gen = Generator(retriever=mock_retriever, config=config)

    prompt_text = gen.prompt.messages[0].prompt.template.lower()

    assert "lead with the specific facts" in prompt_text
    assert "numbers" in prompt_text
    assert "do not open with a general" in prompt_text
    # The original anti-hallucination guards must survive.
    assert "only the provided context" in prompt_text
    assert "i don't know" in prompt_text


def test_prompt_keeps_citation_instruction(mock_retriever):
    config = {"generator": {"model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 64}}
    gen = Generator(retriever=mock_retriever, config=config)

    prompt_text = gen.prompt.messages[0].prompt.template.lower()

    assert "cite" in prompt_text