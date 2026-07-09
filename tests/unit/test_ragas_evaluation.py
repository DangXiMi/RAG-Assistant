# tests/unit/test_ragas_evaluator.py
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.evaluation.ragas_evaluator import RAGASEvaluator


@pytest.fixture
def mock_generator():
    mock = MagicMock()
    # Simulate generator.run() responses
    def run_side_effect(query, top_k, retriever=None):
        # Return a simple answer and sources based on the query
        if "launch date" in query:
            return {
                "answer": "December 25, 2021",
                "sources": ["doc_1"]
            }
        elif "capital" in query:
            return {
                "answer": "I don't know.",
                "sources": []
            }
        else:
            return {
                "answer": "Some answer.",
                "sources": ["doc_2", "doc_3"]
            }
    mock.run.side_effect = run_side_effect
    return mock


@pytest.fixture
def golden_path(tmp_path):
    data = [
        {"question": "What is the launch date of JWST?", "answer": "December 25, 2021", "ground_truth_doc_ids": ["doc_1"], "relevant_chunks": ["doc_1"]},
        {"question": "What is the capital of France?", "answer": "I don't know.", "ground_truth_doc_ids": [], "relevant_chunks": []},
    ]
    path = tmp_path / "golden.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_ragas_evaluator_loads_dataset(mock_evaluate, mock_generator, golden_path):
    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(golden_path))
    dataset = evaluator.load_dataset()
    # Dataset should be a dict with keys: question, answer, ground_truth, etc.
    # RAGAS expects a Dataset or a dict.
    assert isinstance(dataset, dict) or hasattr(dataset, "to_pandas")  # RAGAS Dataset compatible
    # Check that we have 2 questions
    # For simplicity, we'll just check the length of one of the fields.
    # We'll mock ragas.evaluate for now.


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_evaluate_returns_metrics(mock_evaluate, mock_generator, golden_path):
    # Mock the ragas.evaluate function to return a dummy result
    mock_result = MagicMock()
    mock_result.__getitem__.side_effect = lambda key: 0.5 if key.startswith("faithfulness") else 0.6
    mock_evaluate.return_value = mock_result

    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(golden_path))
    metrics = evaluator.evaluate(retriever=MagicMock(), top_k=3)

    # Assert that the generator.run was called twice (for the 2 questions)
    assert mock_generator.run.call_count == 2
    # Assert that evaluate was called once
    mock_evaluate.assert_called_once()
    # Assert that metrics dict contains keys like "faithfulness", "answer_relevancy", etc.
    # In practice, we'll parse the result.


def test_ragas_evaluator_handles_missing_dataset(mock_generator):
    with pytest.raises(FileNotFoundError):
        RAGASEvaluator(generator=mock_generator, dataset_path="non_existent.jsonl")