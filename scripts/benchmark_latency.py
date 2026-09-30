"""Measure end-to-end latency and token usage per retrieval mode.

Produces the baseline that AGENTS.md requires before any optimisation claim,
writing `data/evaluation/latency_summary.csv` (per-mode p50/p95 per stage) and
`data/evaluation/latency_samples.jsonl` (every raw sample).

Usage:
    python scripts/benchmark_latency.py --runs 3
    python scripts/benchmark_latency.py --modes Hybrid,Reranked
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import ingestion_options  # noqa: E402
from src.evaluation.metrics import MetricsRecorder, SUMMARY_FILE  # noqa: E402
from src.ingestion.data_pipeline import load_pipeline  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")

# Questions used for every mode, so the comparison is like-for-like.
BENCHMARK_QUESTIONS = [
    "What is the refund window for a standard item?",
    "What is the shipping lead time for Europe?",
    "Are sale items refundable?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="repetitions per question")
    parser.add_argument("--top-k", type=int, default=3, help="chunks to retrieve")
    parser.add_argument("--modes", default="", help="comma-separated subset of modes")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="queries to run before timing, to avoid measuring cold model loads",
    )
    args = parser.parse_args()

    print("Loading pipeline...", flush=True)
    pipeline = load_pipeline()
    generator = pipeline["generator"]
    retrievers = pipeline["retrievers"]

    selected = list(retrievers)
    if args.modes:
        wanted = [m.strip() for m in args.modes.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in retrievers]
        if unknown:
            print(f"Unknown mode(s): {unknown}. Available: {selected}")
            return 2
        selected = wanted

    # Warm up, so the first measurement is not dominated by lazy model loading.
    if args.warmup:
        print(f"Warming up with {args.warmup} query/queries...", flush=True)
        for _ in range(args.warmup):
            try:
                generator.run(
                    BENCHMARK_QUESTIONS[0],
                    top_k=args.top_k,
                    retriever=retrievers[selected[0]],
                )
            except Exception as exc:
                print(f"  warmup failed: {type(exc).__name__}: {exc}")

    summaries = []
    for mode in selected:
        print(f"\nBenchmarking mode: {mode}", flush=True)
        recorder = MetricsRecorder(mode=mode)

        for run_index in range(args.runs):
            for question in BENCHMARK_QUESTIONS:
                try:
                    result = generator.run(
                        question,
                        top_k=args.top_k,
                        retriever=retrievers[mode],
                        metrics=recorder,
                    )
                    status = "answered" if result.get("answered") else "not-found"
                    print(f"  run {run_index + 1}: {status} :: {question[:52]}")
                except Exception as exc:
                    print(f"  run {run_index + 1}: FAILED {type(exc).__name__}: {exc}")

        summary = recorder.summary()
        summaries.append(summary)
        recorder.save()

        print(f"\n  {mode} summary:")
        for stage, stats in summary["stages"].items():
            print(
                f"    {stage:<10} n={stats['count']:<3} "
                f"mean={stats['mean_ms']:>9.1f}ms "
                f"p50={stats['p50_ms']:>9.1f}ms "
                f"p95={stats['p95_ms']:>9.1f}ms"
            )
        tokens = summary["tokens"]
        print(
            f"    tokens     prompt={tokens['prompt_tokens']} "
            f"completion={tokens['completion_tokens']} "
            f"total={tokens['total_tokens']}"
        )
        print(f"    queries    {summary['queries']} ({summary['not_found']} not-found)")

    print()
    print("=" * 72)
    print("BASELINE COMPARISON (p95 per stage)")
    print("=" * 72)
    stages = sorted({stage for s in summaries for stage in s["stages"]})
    header = f"{'mode':<12}" + "".join(f"{stage:>14}" for stage in stages)
    print(header)
    print("-" * len(header))
    for summary in summaries:
        line = f"{summary['mode']:<12}"
        for stage in stages:
            stats = summary["stages"].get(stage)
            cell = f"{stats['p95_ms']:.0f}ms" if stats else "-"
            line += f"{cell:>14}"
        print(line)

    print()
    print(f"Summary written to {SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
