# Data

This folder is intentionally empty in version control (see `.gitignore`) —
CFPB complaint data is large (millions of rows) and the cleaned/filtered
derivatives are build artifacts, not source.

## Obtaining the raw dataset

1. Download the CFPB Consumer Complaint Database export (CSV) from
   https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Save it as `data/complaints.csv` (override the path via
   `CREDITRUST_RAW_DATA_PATH` if you'd rather keep it elsewhere).

## Regenerating the derived files

```bash
python scripts/run_pipeline.py all
```

This runs, in order:

1. **preprocess** — `data/complaints.csv` → filter by product category →
   drop rows with no narrative → drop exact-duplicate narratives → clean
   text → `data/filtered_complaints.csv`. A data-quality report (null rate,
   duplicate rate, product distribution) is logged before any transformation
   runs, and the stage hard-fails (`DataValidationError`) if required
   columns are missing or the input is empty.
2. **index** — chunk `filtered_complaints.csv` narratives, embed each chunk
   with `sentence-transformers/all-MiniLM-L6-v2`, and persist to the Chroma
   vector store at `vector_store/chroma_index/`.

Both stages are idempotent: `preprocess` skips work if its output already
exists (pass `--force` to rebuild), and every run appends a structured
record (rows in/out, duration, success/failure) to `logs/pipeline_runs.jsonl`.
