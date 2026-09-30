"""Measure answer quality on the demo questions.

Run once before a change and once after, then compare: this is the baseline
comparison AGENTS.md requires before claiming an improvement.

Usage:
    python scripts/measure_answers.py baseline
    python scripts/measure_answers.py after
"""
import json
import sys
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

BASE = "http://127.0.0.1:8010"
OUT_DIR = Path("data/evaluation")

# Each question names phrases that a *good* answer should contain. These are
# taken from the demo corpus, so they are verifiable expectations rather than
# opinions about the wording.
CASES = [
    {
        "question": "What is the refund policy?",
        "mode": "Hybrid",
        "expect": ["30 days"],
        "note": "the brief's own example question",
    },
    {
        "question": "How many days do I have to return a standard item?",
        "mode": "Hybrid",
        "expect": ["30"],
    },
    {
        "question": "What is the shipping lead time for Europe?",
        "mode": "Hybrid",
        "expect": ["5"],
    },
    {
        "question": "Are sale items refundable?",
        "mode": "Hybrid",
        "expect": ["not refundable"],
    },
    {
        "question": "When is support available?",
        "mode": "Hybrid",
        "expect": ["monday"],
    },
    {
        "question": "What is the capital of France?",
        "mode": "Hybrid",
        "expect": ["don't know"],
        "is_refusal": True,
    },
    {
        "question": "What is the refund policy?",
        "mode": "Reranked",
        "expect": ["30 days"],
    },
    {
        "question": "What is the refund policy?",
        "mode": "HyDE",
        "expect": ["30 days"],
    },
]


def ask(question, mode, attempts=3):
    for _ in range(attempts):
        try:
            response = requests.post(
                f"{BASE}/api/v1/query",
                json={"question": question, "user_id": "measure", "mode": mode},
                timeout=900,
            )
        except Exception:
            continue
        if response.status_code == 200:
            return response.json()
    return None


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"

    results = []
    hits = 0

    for case in CASES:
        data = ask(case["question"], case["mode"])
        if data is None:
            results.append({**case, "answer": None, "passed": False})
            print(f"[XX] {case['mode']:<10} {case['question'][:46]}")
            continue

        answer = " ".join(data.get("answer", "").split())
        lowered = answer.lower()
        passed = all(needle.lower() in lowered for needle in case["expect"])
        hits += int(passed)

        mark = "ok" if passed else "XX"
        print(f"[{mark}] {case['mode']:<10} {case['question'][:46]}")
        print(f"       answer: {answer[:150]}")
        if not passed:
            print(f"       expected to contain: {case['expect']}")

        results.append(
            {
                "question": case["question"],
                "mode": case["mode"],
                "answer": answer,
                "expect": case["expect"],
                "passed": passed,
                "contexts": len(data.get("contexts") or []),
                "sources": [r.get("label") for r in (data.get("source_refs") or [])],
            }
        )

    total = len(CASES)
    print()
    print(f"RESULT[{label}]: {hits}/{total} answered with the expected fact")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"answer_quality_{label}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
