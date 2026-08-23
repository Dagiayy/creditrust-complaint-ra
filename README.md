# CrediTrust Complaint RAG

Retrieval-Augmented Generation over consumer financial complaint narratives —
ask a natural-language question, get an answer grounded in real CFPB
complaints plus the exact excerpts it was drawn from.

## 1. What this is and why it exists

Financial institutions receive thousands of free-text complaint narratives.
Reading them one by one to spot emerging problems (e.g. "what's going wrong
with BNPL right now?") doesn't scale. This system lets an analyst, QA lead,
or operations manager ask a question in plain English and get:

- a concise, **grounded** answer (the LLM is instructed to answer only from
  retrieved complaint text, and the pipeline refuses to answer at all if
  retrieval didn't return enough relevant context — see
  [`src/creditrust/rag/pipeline.py`](src/creditrust/rag/pipeline.py))
- the **source excerpts** used to produce that answer, with complaint ID and
  product, for auditability

It runs fully offline/on-prem (local embedding model + local LLM), which
matters for data that may be sensitive.

## 2. Architecture

```
CFPB CSV export
      │  (Extract)
      ▼
┌─────────────────────┐
│ data/validation.py   │  schema check, null/duplicate rate, quality report
└─────────┬────────────┘
          │ (Validate)
          ▼
┌─────────────────────┐
│ data/preprocessing.py│  product filter → dedupe → clean text
└─────────┬────────────┘
          │ (Transform, Load → filtered_complaints.csv)
          ▼
┌─────────────────────┐
│ embeddings/chunking.py│  RecursiveCharacterTextSplitter
│ embeddings/indexer.py │  sentence-transformers/all-MiniLM-L6-v2 → Chroma
└─────────┬────────────┘
          ▼
   vector_store/chroma_index/  (persisted)
          │
          ▼
┌─────────────────────┐        ┌────────────────┐
│ rag/retriever.py     │──────▶│ rag/pipeline.py  │──answer + sources──▶  app.py (Streamlit)
│ rag/llm.py            │──────▶│  RAGPipeline.ask()│                    api/main.py (FastAPI)
└─────────────────────┘        └────────────────┘
```

One `RAGPipeline` implementation is shared by the CLI, the API, and the
Streamlit UI — there is exactly one place retrieval/generation logic lives.
Everything below it (retriever, LLM) is injected, which is what makes it
testable without downloading model weights (`tests/test_rag_pipeline.py`
uses fakes).

### Package layout

```
src/creditrust/
├── config.py            # env-driven Settings, all paths anchored to project root
├── logging_config.py    # structured logging (console + file)
├── data/
│   ├── preprocessing.py # filter / dedupe / clean
│   └── validation.py    # schema + data-quality checks
├── embeddings/
│   ├── chunking.py      # text splitting + metadata sanitization
│   └── indexer.py       # embed + persist to Chroma
├── rag/
│   ├── llm.py            # LLMProvider abstraction (local HF model | mock)
│   ├── retriever.py       # Chroma retriever construction
│   ├── prompts.py         # prompt template + "insufficient context" guardrail
│   └── pipeline.py        # RAGPipeline: retrieve → prompt → generate
└── api/
    ├── main.py            # FastAPI service
    └── schemas.py         # request/response models

scripts/
├── run_pipeline.py         # CLI: preprocess | index | all | ask
└── evaluate_retrieval.py   # retrieval hit-rate@k eval harness

app.py                      # Streamlit UI (presentation only)
tests/                      # pytest — unit tests, no model downloads required
notebooks/eda_preprocessing.ipynb   # EDA, calls into src/creditrust/data
```

## 3. Technologies and why

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small (~90MB), fast on CPU, strong quality/cost tradeoff for semantic search. |
| Vector store | ChromaDB | Embedded, zero-ops, persists to disk — no separate DB service to run for a single-node RAG workload. |
| LLM | Configurable local HF model (default `Qwen/Qwen2.5-0.5B-Instruct`) | Instruction-tuned, small enough for CPU inference, offline-capable. Swappable via `CREDITRUST_LLM_MODEL_NAME` without code changes. |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Battle-tested recursive splitting with configurable overlap; avoided pulling in the rest of a LangChain agent stack. |
| API | FastAPI | Typed request/response models, auto-generated OpenAPI docs, async-ready. |
| UI | Streamlit | Fastest path to a usable internal tool for non-technical staff. |

No Airflow/Dagster/Kafka/Spark/vector-DB-as-a-service, and no
microservices split beyond API/UI — the data volume and single-node
deployment target don't justify that complexity (see `UPGRADE.md` §21 for
the explicit reasoning).

## 4. Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # includes runtime deps + pytest/ruff/black
cp .env.example .env                  # optional — defaults work out of the box
```

Requires Python 3.10+. See `data/README.md` for obtaining the raw dataset.

## 5. Running the pipeline

```bash
# 1. Clean/filter the raw CFPB export (data/complaints.csv -> data/filtered_complaints.csv)
python scripts/run_pipeline.py preprocess

# 2. Chunk, embed, and index into ChromaDB
python scripts/run_pipeline.py index

# ...or both in one shot
python scripts/run_pipeline.py all

# Quick CLI smoke test
python scripts/run_pipeline.py ask "What are common issues with student loans?"
```

Every run is logged to `logs/pipeline_runs.jsonl` (stage, row counts,
duration, success/failure) and `preprocess` is idempotent — it skips work if
`filtered_complaints.csv` already exists (`--force` to rebuild).

## 6. Running the applications

```bash
# Web UI
streamlit run app.py

# API (OpenAPI docs at /docs)
uvicorn creditrust.api.main:app --reload
curl -X POST localhost:8000/api/v1/query -H "Content-Type: application/json" \
     -d '{"question": "Why are customers upset even after paying off debt?"}'
```

Both require the vector index to exist (step 5) — they fail fast with an
actionable error message (not a stack trace) if it doesn't.

### Docker

```bash
docker compose up --build
# API:  http://localhost:8000/docs
# UI:   http://localhost:8501
```

`data/`, `vector_store/`, and `logs/` are bind-mounted so the index survives
container restarts and can be built once, outside Docker, and reused.

## 7. Configuration

All configuration is environment-driven (`src/creditrust/config.py`, backed
by `pydantic-settings`) — see `.env.example` for the full list. Highlights:

| Variable | Default | Purpose |
|---|---|---|
| `CREDITRUST_FILTER_MODE` | `strict` | `strict` \| `expanded` \| `all` product filtering |
| `CREDITRUST_LLM_PROVIDER` | `local` | `local` (real HF model) \| `mock` (canned responses, used in tests) |
| `CREDITRUST_LLM_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Any HF causal-LM repo id |
| `CREDITRUST_TOP_K` | `5` | Chunks retrieved per query |
| `CREDITRUST_API_KEY` | unset | If set, `X-API-Key` header required on `/api/v1/*` |

No secrets are hard-coded anywhere in the codebase; `CREDITRUST_HF_TOKEN` is
only needed for gated HF models.

## 8. Testing

```bash
pytest                          # unit tests — no model downloads, run in <10s
pytest --cov=creditrust         # with coverage
ruff check src scripts tests app.py
black --check src scripts tests app.py
```

The suite covers preprocessing/dedup/cleaning logic, data-quality
validation, chunking + metadata sanitization, config resolution (including
the path-anchoring fix — see `UPGRADE.md`), the RAG pipeline's context
guardrail (via fake retriever/LLM, no network), and the FastAPI routes
(health, auth, 503-when-not-ready) via `TestClient`. CI runs the full suite
with `CREDITRUST_LLM_PROVIDER=mock` on every push.

## 9. Example workflow

1. Ops analyst opens the Streamlit UI, asks *"What complaints relate to
   identity theft?"*
2. `RAGPipeline.ask()` embeds the question, retrieves the top-k most similar
   complaint chunks from Chroma.
3. If total retrieved context is too short to be useful, the pipeline
   returns a fixed "not enough information" answer instead of letting the
   LLM guess — this is deliberate: hallucinated financial-complaint claims
   are worse than an honest "I don't know."
4. Otherwise, the LLM summarizes the retrieved excerpts into a grounded
   3–4 sentence answer.
5. The UI shows the answer plus every source excerpt (complaint ID +
   product), so the analyst can verify the claim against the original text.

## 10. Deployment

- `Dockerfile` builds one image; `docker-compose.yml` runs it twice (API +
  UI) with shared data/vector-store/log volumes.
- `.github/workflows/ci.yml` lints (ruff), format-checks (black), and runs
  the unit test suite (mocked LLM, no downloads) on every push/PR.
- Health checks: `GET /health` reports `vector_store_ready` and the active
  `llm_provider` so an orchestrator can gate traffic until the index exists.

## 11. Monitoring / observability

- Structured logs to console + `logs/app.log` (`CREDITRUST_LOG_JSON=true`
  for machine-parseable logs).
- `logs/pipeline_runs.jsonl` — an append-only run history: what stage ran,
  when, row counts in/out, duration, success/failure.
- `RAGAnswer.latency_seconds` on every query, surfaced in the UI and logged
  server-side, to track end-to-end response time.
- `scripts/evaluate_retrieval.py` — a repeatable retrieval hit-rate@k check
  against a small labeled query set, so a regression in retrieval quality
  after a model/config change shows up as a number, not a vibe.

## 12. Project structure

See §2 above.

## 13. Limitations

- **No labeled relevance judgments.** The retrieval eval
  (`scripts/evaluate_retrieval.py`) uses a product-match proxy for
  relevance, not human-graded qrels — good for catching regressions, not
  for an absolute quality number.
- **Small default LLM.** `Qwen2.5-0.5B-Instruct` runs on CPU but is not as
  capable as a larger hosted model; `CREDITRUST_LLM_MODEL_NAME` is the knob
  to reach for if better answer quality is worth the extra latency/memory.
- **No authentication beyond a single static API key.** Fine for an
  internal tool behind a VPN/reverse proxy; not a substitute for real
  user-level auth/RBAC in a multi-tenant deployment.
- **No automatic drift/quality monitoring in production** — latency is
  logged, but nothing pages anyone if answer quality silently degrades.

## 14. Future improvements

- Human-labeled relevance/answer-quality eval set + periodic scoring.
- Hybrid (keyword + vector) retrieval for queries with specific terms
  (account numbers, exact product names) that pure semantic search can miss.
- Swap the static API key for real auth (OAuth2/JWT) if this moves beyond
  an internal tool.
- Incremental indexing (only embed new/changed complaints) instead of a
  full rebuild, once complaint volume makes that worthwhile.

---
See [`UPGRADE.md`](UPGRADE.md) for the full audit findings and everything
changed from the original prototype.
