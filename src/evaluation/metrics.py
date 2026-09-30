# src/evaluation/metrics.py
"""Latency and token/cost measurement for the RAG pipeline.

AGENTS.md requires every optimisation to have a measured baseline, and the
project previously had no timing anywhere. This module records per-stage
durations and token usage, aggregates them into percentiles, and persists both
raw samples and a summary so a run can be compared against a later one.

Deliberately dependency-free and cheap: timing adds microseconds, so it can
stay enabled in normal operation.
"""
from __future__ import annotations

import json
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/evaluation")
RAW_SAMPLES_FILE = RESULTS_DIR / "latency_samples.jsonl"
SUMMARY_FILE = RESULTS_DIR / "latency_summary.csv"

# Approximate counts per 1000 tokens, used only to report a rough cost figure.
# Local Ollama inference is free at the margin, so this is informational and
# defaults to zero rather than inventing a number.
DEFAULT_PRICE_PER_1K_TOKENS = 0.0


@dataclass
class StageSample:
    """One timed execution of a pipeline stage."""

    stage: str
    seconds: float
    mode: str = "unknown"
    question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsRecorder:
    """Collects stage timings and token usage for a benchmark run.

    Usage:
        recorder = MetricsRecorder(mode="Hybrid")
        with recorder.stage("retrieve"):
            docs = retriever.search(query)
        recorder.add_tokens(prompt=120, completion=45)
    """

    def __init__(self, mode: str = "unknown", price_per_1k_tokens: float = DEFAULT_PRICE_PER_1K_TOKENS):
        self.mode = mode
        self.price_per_1k_tokens = price_per_1k_tokens
        self.samples: list[StageSample] = []
        self.token_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.query_count = 0
        self.not_found_count = 0

    @contextmanager
    def stage(self, stage: str, question: str = "", **metadata) -> Iterator[None]:
        """Time a block and record it under `stage`."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.samples.append(
                StageSample(
                    stage=stage,
                    seconds=elapsed,
                    mode=self.mode,
                    question=question[:200],
                    metadata=metadata,
                )
            )

    def add_stage_sample(self, stage: str, seconds: float, question: str = "", **metadata) -> None:
        """Record a duration measured outside a `with recorder.stage(...)` block."""
        self.samples.append(
            StageSample(
                stage=stage,
                seconds=seconds,
                mode=self.mode,
                question=question[:200],
                metadata=metadata,
            )
        )

    def add_tokens(self, prompt: int = 0, completion: int = 0) -> None:
        """Accumulate token usage reported by the LLM."""
        self.token_totals["prompt_tokens"] += int(prompt or 0)
        self.token_totals["completion_tokens"] += int(completion or 0)
        self.token_totals["total_tokens"] += int(prompt or 0) + int(completion or 0)

    def record_query(self, answered: bool = True) -> None:
        """Count one answered query and whether it was grounded."""
        self.query_count += 1
        if not answered:
            self.not_found_count += 1

    # --- aggregation --------------------------------------------------------

    def stage_stats(self) -> dict[str, dict[str, float]]:
        """Per-stage count, mean, p50 and p95 latency in milliseconds."""
        grouped: dict[str, list[float]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.stage, []).append(sample.seconds)

        stats: dict[str, dict[str, float]] = {}
        for stage, values in grouped.items():
            stats[stage] = {
                "count": len(values),
                "mean_ms": round(statistics.fmean(values) * 1000, 2),
                "p50_ms": round(_percentile(values, 50) * 1000, 2),
                "p95_ms": round(_percentile(values, 95) * 1000, 2),
                "max_ms": round(max(values) * 1000, 2),
            }
        return stats

    def cost_estimate(self) -> float:
        """Rough cost from token totals and the configured price."""
        return round(
            self.token_totals["total_tokens"] / 1000.0 * self.price_per_1k_tokens, 6
        )

    def summary(self) -> dict[str, Any]:
        """Full result for one mode, ready to serialise."""
        return {
            "mode": self.mode,
            "queries": self.query_count,
            "not_found": self.not_found_count,
            "stages": self.stage_stats(),
            "tokens": dict(self.token_totals),
            "estimated_cost": self.cost_estimate(),
        }

    def rows(self) -> list[dict[str, Any]]:
        """Flattened per-stage rows, convenient for a CSV summary."""
        rows = []
        for stage, stats in self.stage_stats().items():
            row = {"mode": self.mode, "stage": stage}
            row.update(stats)
            rows.append(row)
        return rows

    # --- persistence --------------------------------------------------------

    def save(self, raw_path: Path = RAW_SAMPLES_FILE,
             summary_path: Path = SUMMARY_FILE) -> tuple[Path, Path]:
        """Append raw samples to JSONL and write the per-mode summary CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with raw_path.open("a", encoding="utf-8") as handle:
            for sample in self.samples:
                handle.write(
                    json.dumps(
                        {
                            "stage": sample.stage,
                            "seconds": sample.seconds,
                            "mode": sample.mode,
                            "question": sample.question,
                            **sample.metadata,
                        }
                    )
                    + "\n"
                )

        import csv

        existing: list[dict[str, Any]] = []
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                existing = [
                    row for row in csv.DictReader(handle) if row.get("mode") != self.mode
                ]

        rows = existing + [str_value(r) for r in self.rows()]
        fieldnames = sorted({key for row in rows for key in row})
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Latency samples appended to %s", raw_path)
        logger.info("Latency summary written to %s", summary_path)
        return raw_path, summary_path


def _percentile(values: list[float], percent: float) -> float:
    """Nearest-rank percentile, adequate for the sample sizes here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def str_value(row: dict[str, Any]) -> dict[str, Any]:
    """Stringify row values so mixed types can share a CSV."""
    return {key: ("" if value is None else str(value)) for key, value in row.items()}


def extract_token_usage(message: Any) -> tuple[int, int]:
    """Pull prompt/completion token counts out of a LangChain message.

    Ollama reports these under `response_metadata`; other providers differ, so
    a missing value yields 0 rather than failing a query.
    """
    metadata = getattr(message, "response_metadata", None) or {}
    usage = getattr(message, "usage_metadata", None) or {}

    prompt = (
        usage.get("input_tokens")
        or metadata.get("prompt_eval_count")
        or metadata.get("prompt_tokens")
        or 0
    )
    completion = (
        usage.get("output_tokens")
        or metadata.get("eval_count")
        or metadata.get("completion_tokens")
        or 0
    )
    return int(prompt), int(completion)


def compare_summaries(baseline_path: Path, candidate_path: Path) -> list[dict[str, Any]]:
    """Diff two latency summaries by mode and stage.

    Returns rows with absolute and percentage change, so an optimisation can be
    reported as a measured improvement or a regression.
    """
    import csv

    def load(path: Path) -> dict[tuple[str, str], float]:
        data: dict[tuple[str, str], float] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    data[(row["mode"], row["stage"])] = float(row["p95_ms"])
                except (KeyError, TypeError, ValueError):
                    continue
        return data

    baseline = load(baseline_path)
    candidate = load(candidate_path)

    changes = []
    for key, old in baseline.items():
        new = candidate.get(key)
        if new is None:
            continue
        delta = new - old
        changes.append(
            {
                "mode": key[0],
                "stage": key[1],
                "baseline_p95_ms": round(old, 2),
                "candidate_p95_ms": round(new, 2),
                "delta_ms": round(delta, 2),
                "change_pct": round((delta / old * 100), 1) if old else 0.0,
            }
        )
    return changes
