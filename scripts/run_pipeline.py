#!/usr/bin/env python
"""CLI entrypoint for the data + embedding pipeline.

Stages:  Extract -> Validate -> Transform -> Load -> Verify

Usage:
    python scripts/run_pipeline.py preprocess   # raw CSV -> filtered_complaints.csv
    python scripts/run_pipeline.py index        # filtered_complaints.csv -> Chroma index
    python scripts/run_pipeline.py all          # both, in order
    python scripts/run_pipeline.py ask "What are common issues with student loans?"

Every run appends a JSON line to `logs/pipeline_runs.jsonl` recording what
ran, when, how many rows were processed, whether it succeeded, and how long
it took — the minimum viable observability for "what ran and did it work?".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from creditrust.config import get_settings  # noqa: E402
from creditrust.data.preprocessing import load_data as load_raw_data  # noqa: E402
from creditrust.data.preprocessing import preprocess_complaints, save_cleaned_data  # noqa: E402
from creditrust.data.validation import DataValidationError, validate_or_raise  # noqa: E402
from creditrust.embeddings.indexer import build_vector_store  # noqa: E402
from creditrust.logging_config import get_logger  # noqa: E402
from creditrust.rag.pipeline import RAGPipeline  # noqa: E402

logger = get_logger(__name__)


def _record_run(stage: str, status: str, extra: dict) -> None:
    settings = get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "status": status, "timestamp": time.time(), **extra}
    with open(settings.pipeline_run_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_preprocess(args: argparse.Namespace) -> None:
    settings = get_settings()
    start = time.perf_counter()
    raw_path = args.input or settings.raw_data_path
    out_path = args.output or settings.filtered_data_path

    if out_path.exists() and not args.force:
        logger.info("%s already exists — skipping (use --force to rebuild).", out_path)
        _record_run("preprocess", "skipped", {"reason": "output_exists", "output": str(out_path)})
        return

    try:
        df = load_raw_data(str(raw_path))
        report = validate_or_raise(df)
        logger.info("Raw data quality report: %s", report.to_dict())

        cleaned = preprocess_complaints(
            df,
            apply_min_word_filter=settings.apply_min_word_filter,
            min_words=settings.min_narrative_words,
            filter_mode=settings.filter_mode,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_cleaned_data(cleaned, str(out_path))

        duration = time.perf_counter() - start
        _record_run(
            "preprocess",
            "success",
            {"input_rows": len(df), "output_rows": len(cleaned), "duration_seconds": duration},
        )
        logger.info("Preprocessing complete in %.1fs: %d -> %d rows", duration, len(df), len(cleaned))
    except (DataValidationError, FileNotFoundError) as exc:
        _record_run("preprocess", "failed", {"error": str(exc)})
        logger.error("Preprocessing failed: %s", exc)
        raise SystemExit(1) from exc


def run_index(args: argparse.Namespace) -> None:
    settings = get_settings()
    start = time.perf_counter()
    filtered_path = args.input or settings.filtered_data_path

    if not filtered_path.exists():
        msg = f"{filtered_path} not found. Run the 'preprocess' stage first."
        _record_run("index", "failed", {"error": msg})
        logger.error(msg)
        raise SystemExit(1)

    try:
        df = load_raw_data(str(filtered_path))
        count = build_vector_store(df, persist_directory=settings.vector_store_dir)
        duration = time.perf_counter() - start
        _record_run(
            "index",
            "success",
            {"input_rows": len(df), "chunks_indexed": count, "duration_seconds": duration},
        )
        logger.info("Indexing complete in %.1fs: %d chunks indexed", duration, count)
    except Exception as exc:  # noqa: BLE001
        _record_run("index", "failed", {"error": str(exc)})
        logger.exception("Indexing failed")
        raise SystemExit(1) from exc


def run_ask(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline.from_settings()
    result = pipeline.ask(args.question)
    print("\n--- Answer ---")
    print(result.answer)
    print(
        f"\n(answered in {result.latency_seconds:.2f}s, sufficient_context={result.had_sufficient_context})"
    )
    if result.sources:
        print("\n--- Sources ---")
        for s in result.sources:
            print(f"- [{s.complaint_id}] {s.product}: {s.excerpt[:150]}...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess", help="Clean/filter the raw complaint export.")
    p_pre.add_argument("--input", type=Path, default=None)
    p_pre.add_argument("--output", type=Path, default=None)
    p_pre.add_argument("--force", action="store_true", help="Rebuild even if output already exists.")
    p_pre.set_defaults(func=run_preprocess)

    p_idx = sub.add_parser("index", help="Chunk, embed, and index the cleaned data into Chroma.")
    p_idx.add_argument("--input", type=Path, default=None)
    p_idx.set_defaults(func=run_index)

    p_all = sub.add_parser("all", help="Run preprocess then index.")
    p_all.add_argument("--force", action="store_true")
    p_all.set_defaults(func=lambda a: (run_preprocess(a), run_index(a)), input=None, output=None)

    p_ask = sub.add_parser("ask", help="Ask a single question via the CLI (for smoke-testing).")
    p_ask.add_argument("question", type=str)
    p_ask.set_defaults(func=run_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
