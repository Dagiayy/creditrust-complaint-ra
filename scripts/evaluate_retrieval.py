#!/usr/bin/env python
"""Lightweight retrieval evaluation harness.

Runs a small labeled query set against the live vector index and reports,
per query: whether any retrieved chunk's `product` metadata matches the
expected product (a cheap proxy for retrieval relevance that needs no manual
relevance judgments), plus mean retrieval latency.

This is intentionally simple — it is not a substitute for human-labeled
relevance judgments (e.g. via MRR/nDCG on a curated qrels set), but it turns
"does retrieval still work after a change" into a single command instead of
manual spot-checking, and it's the natural place to grow a real eval set.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --eval-set path/to/eval_set.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrust.config import get_settings  # noqa: E402
from creditrust.logging_config import get_logger  # noqa: E402
from creditrust.rag.retriever import get_retriever  # noqa: E402

logger = get_logger(__name__)

DEFAULT_EVAL_SET = [
    {"question": "What are common issues with credit card billing?", "expected_product": "Credit card"},
    {"question": "Why do personal loan customers complain about fees?", "expected_product": "Personal loan"},
    {
        "question": "What problems do people have with their savings accounts?",
        "expected_product": "Savings account",
    },
    {
        "question": "Are there complaints about Buy Now Pay Later services?",
        "expected_product": "Buy Now, Pay Later (BNPL)",
    },
    {"question": "What issues arise with money transfers?", "expected_product": "Money transfers"},
]


def load_eval_set(path: Path | None) -> list[dict]:
    if path is None:
        return DEFAULT_EVAL_SET
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(eval_set: list[dict], top_k: int) -> dict:
    settings = get_settings()
    retriever = get_retriever(settings.vector_store_dir, top_k=top_k)

    results = []
    for case in eval_set:
        start = time.perf_counter()
        docs = retriever.invoke(case["question"])
        latency = time.perf_counter() - start
        products_retrieved = [d.metadata.get("product") for d in docs]
        hit = case["expected_product"] in products_retrieved
        results.append(
            {
                "question": case["question"],
                "expected_product": case["expected_product"],
                "products_retrieved": products_retrieved,
                "hit": hit,
                "latency_seconds": round(latency, 4),
            }
        )

    hit_rate = sum(r["hit"] for r in results) / len(results) if results else 0.0
    mean_latency = sum(r["latency_seconds"] for r in results) / len(results) if results else 0.0
    return {"hit_rate_at_k": hit_rate, "mean_latency_seconds": round(mean_latency, 4), "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set", type=Path, default=None, help="JSON file: [{question, expected_product}, ...]"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    report = evaluate(eval_set, top_k=args.top_k)

    print(json.dumps(report, indent=2))
    logger.info(
        "Retrieval eval: hit_rate@k=%.2f, mean_latency=%.3fs",
        report["hit_rate_at_k"],
        report["mean_latency_seconds"],
    )


if __name__ == "__main__":
    main()
