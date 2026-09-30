# tests/unit/test_ragas_evaluator.py
import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.evaluation.ragas_evaluator import RAGASEvaluator


@pytest.fixture
def mock_generator():
    """Mock whose `run` returns the same keys the real Generator does.

    RAGAS needs `answer` and `contexts`; an incomplete dict would fail deep
    inside the evaluator rather than at the boundary.
    """
    mock = MagicMock()

    def run_side_effect(query, top_k=5, retriever=None):
        if "launch date" in query:
            return {
                "answer": "December 25, 2021",
                "contexts": ["JWST was launched on December 25, 2021."],
                "sources": ["[1] jwst.pdf, p.1"],
                "source_refs": [{"ref": 1, "source": "jwst.pdf", "location": "p.1"}],
                "retrieved": 1,
                "answered": True,
            }
        elif "capital" in query:
            return {
                "answer": "I don't know.",
                "contexts": [],
                "sources": [],
                "source_refs": [],
                "retrieved": 0,
                "answered": False,
            }
        return {
            "answer": "Some answer.",
            "contexts": ["Some retrieved context."],
            "sources": ["[1] doc.pdf, p.1"],
            "source_refs": [{"ref": 1, "source": "doc.pdf", "location": "p.1"}],
            "retrieved": 1,
            "answered": True,
        }

    mock.run.side_effect = run_side_effect
    return mock


@pytest.fixture
def golden_path(tmp_path):
    # Mirrors the real data/evaluation/golden.jsonl schema: the evaluator reads
    # `ground_truth_chunks` and joins it into the reference text.
    data = [
        {
            "question": "What is the launch date of JWST?",
            "answer": "December 25, 2021",
            "ground_truth_chunks": ["JWST was launched on December 25, 2021."],
            "relevant_chunks": ["doc_1"],
        },
        {
            "question": "What is the capital of France?",
            "answer": "I don't know.",
            "ground_truth_chunks": ["The documents do not state the capital of France."],
            "relevant_chunks": [],
        },
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

    assert isinstance(dataset, list)
    assert len(dataset) == 2
    assert dataset[0]["question"] == "What is the launch date of JWST?"
    # ground_truth_chunks must be joined into the reference text RAGAS compares
    # against; context_recall is meaningless if this comes out empty.
    assert dataset[0]["ground_truth"] == "JWST was launched on December 25, 2021."


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_load_dataset_accepts_plain_ground_truth(mock_evaluate, mock_generator, tmp_path):
    path = tmp_path / "alt.jsonl"
    path.write_text(
        json.dumps({"question": "Q?", "ground_truth": "A reference passage."}) + "\n",
        encoding="utf-8",
    )

    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(path))

    assert evaluator.load_dataset()[0]["ground_truth"] == "A reference passage."


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_load_dataset_skips_records_without_reference_text(
    mock_evaluate, mock_generator, tmp_path
):
    """An empty ground_truth would silently corrupt context_recall."""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps({"question": "Has refs", "ground_truth_chunks": ["text"]}) + "\n"
        + json.dumps({"question": "No refs", "ground_truth_chunks": []}) + "\n",
        encoding="utf-8",
    )

    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(path))
    data = evaluator.load_dataset()

    assert [r["question"] for r in data] == ["Has refs"]


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_load_dataset_rejects_a_record_with_no_question(
    mock_evaluate, mock_generator, tmp_path
):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"ground_truth_chunks": ["x"]}) + "\n", encoding="utf-8")

    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(path))

    with pytest.raises(ValueError, match="question"):
        evaluator.load_dataset()


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_evaluate_returns_metrics(mock_evaluate, mock_generator, golden_path, tmp_path):
    # Mock the ragas.evaluate function to return a dummy result
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = pd.DataFrame(
        [
            {
                "question": "What is the launch date of JWST?",
                "faithfulness": 0.5,
                "answer_relevancy": 0.6,
                "context_precision": 0.7,
                "context_recall": 0.8,
            }
        ]
    )
    mock_evaluate.return_value = mock_result

    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(golden_path))
    # save_per_sample=False: this is a mocked run, so no real scores exist.
    metrics = evaluator.evaluate(retriever=MagicMock(), top_k=3, save_per_sample=False)

    # Assert that the generator.run was called twice (for the 2 questions)
    assert mock_generator.run.call_count == 2
    # Assert that evaluate was called once
    mock_evaluate.assert_called_once()
    assert "faithfulness" in metrics["aggregated"]
    assert metrics["per_sample"]


def test_ragas_evaluator_handles_missing_dataset(mock_generator, tmp_path):
    """Fails fast, before the slow judge models are constructed."""
    missing = tmp_path / "non_existent.jsonl"

    with pytest.raises(FileNotFoundError) as excinfo:
        RAGASEvaluator(generator=mock_generator, dataset_path=str(missing))

    assert "non_existent.jsonl" in str(excinfo.value)


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_save_per_sample_writes_one_row_per_question(
    mock_evaluate, mock_generator, golden_path, tmp_path
):
    """Item 14: a low aggregate must be traceable to specific questions."""
    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(golden_path))
    destination = tmp_path / "per_sample.csv"

    per_sample = [
        {
            "question": "What is the launch date of JWST?",
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 1.0,
            "context_recall": 0.9,
        },
        {
            "question": "What is the capital of France?",
            "faithfulness": 0.1,
            "answer_relevancy": 0.2,
            "context_precision": 0.0,
            "context_recall": 0.0,
        },
    ]

    written = evaluator.save_per_sample(per_sample, path=str(destination))

    assert written == destination
    frame = pd.read_csv(destination)
    assert len(frame) == 2
    # The failing question is identifiable, which is the point of the file.
    worst = frame.sort_values("faithfulness").iloc[0]
    assert worst["question"] == "What is the capital of France?"


@patch("src.evaluation.ragas_evaluator.evaluate")
def test_save_per_sample_creates_the_directory(
    mock_evaluate, mock_generator, golden_path, tmp_path
):
    evaluator = RAGASEvaluator(generator=mock_generator, dataset_path=str(golden_path))
    destination = tmp_path / "nested" / "dir" / "per_sample.csv"

    evaluator.save_per_sample([{"question": "q", "faithfulness": 1.0}], path=str(destination))

    assert destination.exists()