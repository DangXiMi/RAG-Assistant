# tests/unit/test_metrics.py
"""Tests for latency and token measurement."""
import csv
import json
import time

import pytest

from src.evaluation.metrics import (
    MetricsRecorder,
    compare_summaries,
    extract_token_usage,
)


class FakeMessage:
    def __init__(self, response_metadata=None, usage_metadata=None):
        self.response_metadata = response_metadata or {}
        self.usage_metadata = usage_metadata or {}


# --- stage timing -----------------------------------------------------------

def test_stage_records_one_sample():
    recorder = MetricsRecorder(mode="Hybrid")

    with recorder.stage("retrieve", question="q"):
        pass

    assert len(recorder.samples) == 1
    assert recorder.samples[0].stage == "retrieve"
    assert recorder.samples[0].mode == "Hybrid"
    assert recorder.samples[0].seconds >= 0


def test_stage_records_a_measurable_duration():
    recorder = MetricsRecorder()

    with recorder.stage("generate"):
        time.sleep(0.05)

    assert recorder.samples[0].seconds >= 0.04


def test_stage_records_even_when_the_block_raises():
    recorder = MetricsRecorder()

    with pytest.raises(ValueError):
        with recorder.stage("retrieve"):
            raise ValueError("boom")

    assert len(recorder.samples) == 1


def test_manual_stage_sample_can_be_added():
    recorder = MetricsRecorder()

    recorder.add_stage_sample("total", 1.25, question="q")

    assert recorder.samples[0].stage == "total"
    assert recorder.samples[0].seconds == 1.25


def test_question_is_truncated_for_storage():
    recorder = MetricsRecorder()

    with recorder.stage("retrieve", question="x" * 500):
        pass

    assert len(recorder.samples[0].question) == 200


# --- aggregation ------------------------------------------------------------

def test_stage_stats_reports_percentiles_in_ms():
    recorder = MetricsRecorder()
    for seconds in (0.1, 0.2, 0.3, 0.4):
        recorder.add_stage_sample("retrieve", seconds)

    stats = recorder.stage_stats()["retrieve"]

    assert stats["count"] == 4
    assert stats["mean_ms"] == pytest.approx(250.0, abs=1.0)
    assert stats["p50_ms"] == pytest.approx(250.0, abs=5.0)
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert stats["max_ms"] == pytest.approx(400.0, abs=1.0)


def test_stage_stats_groups_by_stage():
    recorder = MetricsRecorder()
    recorder.add_stage_sample("retrieve", 0.1)
    recorder.add_stage_sample("generate", 0.5)

    stats = recorder.stage_stats()

    assert set(stats) == {"retrieve", "generate"}
    assert stats["generate"]["count"] == 1


def test_single_sample_percentile_is_the_sample():
    recorder = MetricsRecorder()
    recorder.add_stage_sample("retrieve", 0.25)

    stats = recorder.stage_stats()["retrieve"]

    assert stats["p50_ms"] == stats["p95_ms"] == pytest.approx(250.0, abs=1.0)


def test_stage_stats_is_empty_without_samples():
    assert MetricsRecorder().stage_stats() == {}


# --- tokens and cost --------------------------------------------------------

def test_add_tokens_accumulates():
    recorder = MetricsRecorder()

    recorder.add_tokens(prompt=100, completion=20)
    recorder.add_tokens(prompt=50, completion=10)

    assert recorder.token_totals["prompt_tokens"] == 150
    assert recorder.token_totals["completion_tokens"] == 30
    assert recorder.token_totals["total_tokens"] == 180


def test_add_tokens_tolerates_none():
    recorder = MetricsRecorder()

    recorder.add_tokens(prompt=None, completion=None)

    assert recorder.token_totals["total_tokens"] == 0


def test_cost_is_zero_when_priced_at_zero():
    """Local Ollama inference is free, so no fake cost is reported."""
    recorder = MetricsRecorder()
    recorder.add_tokens(prompt=1000, completion=1000)

    assert recorder.cost_estimate() == 0.0


def test_cost_uses_the_configured_price():
    recorder = MetricsRecorder(price_per_1k_tokens=0.002)
    recorder.add_tokens(prompt=1000, completion=0)

    assert recorder.cost_estimate() == pytest.approx(0.002)


def test_query_counting_tracks_not_found():
    recorder = MetricsRecorder()

    recorder.record_query(answered=True)
    recorder.record_query(answered=False)

    assert recorder.query_count == 2
    assert recorder.not_found_count == 1


# --- extract_token_usage ----------------------------------------------------

def test_extracts_ollama_token_counts():
    message = FakeMessage(response_metadata={"prompt_eval_count": 120, "eval_count": 45})

    assert extract_token_usage(message) == (120, 45)


def test_extracts_openai_style_usage_metadata():
    message = FakeMessage(usage_metadata={"input_tokens": 30, "output_tokens": 7})

    assert extract_token_usage(message) == (30, 7)


def test_missing_token_metadata_yields_zero():
    assert extract_token_usage(FakeMessage()) == (0, 0)


def test_token_extraction_never_raises_on_a_bare_object():
    class Bare:
        pass

    assert extract_token_usage(Bare()) == (0, 0)


# --- persistence ------------------------------------------------------------

def test_save_writes_raw_samples_and_summary(tmp_path):
    raw = tmp_path / "samples.jsonl"
    summary = tmp_path / "summary.csv"

    recorder = MetricsRecorder(mode="Hybrid")
    recorder.add_stage_sample("retrieve", 0.12, question="q")
    recorder.add_tokens(prompt=10, completion=5)
    recorder.record_query(answered=True)

    recorder.save(raw_path=raw, summary_path=summary)

    lines = raw.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["stage"] == "retrieve"

    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["mode"] == "Hybrid"
    assert rows[0]["stage"] == "retrieve"


def test_save_replaces_rows_for_the_same_mode(tmp_path):
    raw = tmp_path / "samples.jsonl"
    summary = tmp_path / "summary.csv"

    first = MetricsRecorder(mode="Hybrid")
    first.add_stage_sample("retrieve", 0.1)
    first.save(raw_path=raw, summary_path=summary)

    second = MetricsRecorder(mode="Hybrid")
    second.add_stage_sample("retrieve", 0.3)
    second.save(raw_path=raw, summary_path=summary)

    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # One row per (mode, stage), not two accumulated Hybrid rows.
    assert len([r for r in rows if r["mode"] == "Hybrid"]) == 1


def test_save_keeps_other_modes(tmp_path):
    raw = tmp_path / "samples.jsonl"
    summary = tmp_path / "summary.csv"

    hybrid = MetricsRecorder(mode="Hybrid")
    hybrid.add_stage_sample("retrieve", 0.1)
    hybrid.save(raw_path=raw, summary_path=summary)

    hyde = MetricsRecorder(mode="HyDE")
    hyde.add_stage_sample("retrieve", 0.2)
    hyde.save(raw_path=raw, summary_path=summary)

    with summary.open(encoding="utf-8", newline="") as handle:
        modes = {row["mode"] for row in csv.DictReader(handle)}
    assert modes == {"Hybrid", "HyDE"}


def test_save_appends_raw_samples(tmp_path):
    raw = tmp_path / "samples.jsonl"
    summary = tmp_path / "summary.csv"

    for value in (0.1, 0.2):
        recorder = MetricsRecorder(mode="Hybrid")
        recorder.add_stage_sample("retrieve", value)
        recorder.save(raw_path=raw, summary_path=summary)

    assert len(raw.read_text(encoding="utf-8").strip().splitlines()) == 2


# --- baseline comparison ----------------------------------------------------

def test_compare_summaries_reports_improvement(tmp_path):
    raw = tmp_path / "s.jsonl"
    base = tmp_path / "base.csv"
    cand = tmp_path / "cand.csv"

    recorder = MetricsRecorder(mode="Hybrid")
    for value in (0.4, 0.5):
        recorder.add_stage_sample("generate", value)
    recorder.save(raw_path=raw, summary_path=base)

    faster = MetricsRecorder(mode="Hybrid")
    for value in (0.1, 0.12):
        faster.add_stage_sample("generate", value)
    faster.save(raw_path=raw, summary_path=cand)

    changes = compare_summaries(base, cand)

    assert len(changes) == 1
    assert changes[0]["delta_ms"] < 0
    assert changes[0]["change_pct"] < 0


def test_compare_summaries_reports_regression(tmp_path):
    raw = tmp_path / "s.jsonl"
    base = tmp_path / "base.csv"
    cand = tmp_path / "cand.csv"

    recorder = MetricsRecorder(mode="Hybrid")
    for value in (0.1, 0.12):
        recorder.add_stage_sample("generate", value)
    recorder.save(raw_path=raw, summary_path=base)

    slower = MetricsRecorder(mode="Hybrid")
    for value in (0.5, 0.6):
        slower.add_stage_sample("generate", value)
    slower.save(raw_path=raw, summary_path=cand)

    changes = compare_summaries(base, cand)

    assert changes[0]["delta_ms"] > 0
    assert changes[0]["change_pct"] > 0


def test_compare_summaries_ignores_missing_stages(tmp_path):
    raw = tmp_path / "s.jsonl"
    base = tmp_path / "base.csv"
    cand = tmp_path / "cand.csv"

    recorder = MetricsRecorder(mode="Hybrid")
    recorder.add_stage_sample("retrieve", 0.1)
    recorder.add_stage_sample("generate", 0.2)
    recorder.save(raw_path=raw, summary_path=base)

    partial = MetricsRecorder(mode="Hybrid")
    partial.add_stage_sample("retrieve", 0.1)
    partial.save(raw_path=raw, summary_path=cand)

    changes = compare_summaries(base, cand)

    assert [c["stage"] for c in changes] == ["retrieve"]
