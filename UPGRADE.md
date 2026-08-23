# Upgrade Log — Prototype → Production-Grade RAG System

This document records the full audit of the original `creditrust-complaint-ra`
prototype and everything changed to bring it to production quality. It is
organized as: what was found, what was done about it, and what's still open.

## 1. Original state (as received)

A 4-stage RAG prototype: `notebooks/eda_preprocessing.ipynb` (cleaning) →
`src/chunking_embedding.py` (embed + index) → `src/rag_pipeline.py` (CLI
Q&A) → `app.py` (Streamlit UI). `src/interface.py` was an empty stub.
`requirements.txt` listed 9 unpinned packages. `.github/workflows/ci.yml`
existed but only printed the Python version. No tests, no Docker, no `.env`
handling, no logging (only `print`), no data validation.

## 2. Critical bugs found and fixed (P0)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | `app.py` imported `from src.rag_core import ask_question` — that module never existed anywhere in the repo (only `src/rag_pipeline.py`, with a different API). | **The Streamlit app could not start at all** — `ModuleNotFoundError` on launch. | Rebuilt `app.py` to import `RAGPipeline` from the new `creditrust.rag.pipeline` module, shared with the API and CLI. |
| 2 | `VECTOR_STORE_DIR = "../vector_store/chroma_index"` (hardcoded relative path) in both `chunking_embedding.py` and `rag_pipeline.py`, while the README instructs running these scripts from the repo root (`python src/chunking_embedding.py`). | Resolves to a directory **outside the repository** when run as documented; would silently write/read the wrong location depending on CWD. | `src/creditrust/config.py` anchors every path to `PROJECT_ROOT = Path(__file__).resolve().parents[2]`, so behavior is identical regardless of CWD or entrypoint (CLI, API, Streamlit, tests). Covered by `tests/test_config.py::test_relative_path_override_is_anchored_to_project_root`. |
| 3 | `.gitignore` was corrupted: its last line mixed CRLF text with UTF-16-encoded bytes (`.cursorindexingignorev\0e\0c\0t\0o\0r\0...`), merging `.cursorindexingignore` and `vector_store/` into one unreadable, non-matching line. | `vector_store/` (a multi-GB Chroma index) and raw/derived data CSVs were **not actually ignored** — a `git add .` would have committed them. | Rewrote `.gitignore` clean (UTF-8, LF-consistent) and added explicit rules for `data/*.csv`, `vector_store/`, `logs/`. |
| 4 | `src/interface.py` — 0 bytes, imported by nothing, referenced by nothing. | Dead file, confusing to a new contributor (README's Task 4 already points at `app.py`). | Removed. |
| 5 | `requirements.txt` had no version pins and was missing packages the code actually imports (`langchain_community`, `langchain_huggingface`, `langchain_chroma`, `tqdm` was listed but under-specified). | Fresh installs are non-reproducible and can pull breaking major versions; some imports would fail outright depending on what `pip` resolves. | Re-pinned with floor versions (`requirements.txt`), split dev tooling into `requirements-dev.txt`. |
| 6 | `notebooks/eda_preprocessing.ipynb` had 3 cells, two of which referenced an undefined `df` (never loaded in the notebook) — the numbers in `notebooks/report.md` (82,164 / 1.27M cleaned complaints) are **not reproducible** from the notebook as shipped. | Reproducibility gap — the documented EDA can't actually be re-run. | Rebuilt the notebook to load data and call the same `creditrust.data.preprocessing` functions the CLI pipeline uses, so notebook and pipeline can't drift apart, and the numbers in `report.md` become reproducible by re-running it. |

## 3. Architecture

**Before:** four standalone scripts with duplicated constants (chunk size,
model names, paths) and no shared abstractions — `app.py` and
`rag_pipeline.py` each had their own copy of the retrieval+generation logic
(and only one of them actually ran).

**After:** an installable `src/creditrust/` package with clear layers —
`config` → `data` (ingest/validate/clean) → `embeddings` (chunk/index) →
`rag` (retrieve/prompt/generate) → `api`/`app.py` (thin presentation). One
`RAGPipeline` class, constructed via dependency injection
(`retriever`, `llm_provider`), is shared by the CLI, the FastAPI service,
and the Streamlit app. See `README.md` §2 for the full diagram.

This is a modular monolith, deliberately — see §10 below for why no
microservices/Airflow/Kafka/Spark were introduced.

## 4. Data engineering improvements

- **Validation gate** (`data/validation.py`): required-column check, null-
  narrative rate, duplicate-row/-narrative rate, product distribution — run
  automatically before every `preprocess` stage, with a hard failure
  (`DataValidationError`) on missing columns or an empty dataset instead of
  a confusing downstream `KeyError`.
- **Duplicate narrative removal** — the original pipeline had no
  deduplication step at all; exact-duplicate narratives (common with
  bulk/templated CFPB submissions) were inflating the index with redundant
  chunks. Added `remove_duplicate_narratives`.
- **Idempotent, observable pipeline runner** (`scripts/run_pipeline.py`):
  `preprocess`/`index`/`all`/`ask` subcommands; `preprocess` skips work if
  its output already exists (`--force` to override); every run appends a
  structured record (stage, row counts, duration, success/failure) to
  `logs/pipeline_runs.jsonl` — the minimum viable "what ran, when, did it
  work" audit trail.
- **`data/README.md`** documents data provenance (where to get the CFPB
  export) and exactly how to regenerate every derived artifact — none of
  that was previously written down anywhere.

## 5. ML / retrieval improvements

- **LLM provider abstraction** (`rag/llm.py`): `LLMProvider` ABC with a
  `LocalHFProvider` (lazy-loaded HF pipeline, model swappable via
  `CREDITRUST_LLM_MODEL_NAME`) and a `MockLLMProvider` used in tests/offline
  demos. The original code hard-imported and loaded a 2.6GB model
  (`tiiuae/falcon-rw-1b`) at **module import time** in `rag_pipeline.py` —
  meaning even importing that file for testing triggered a multi-GB
  download. That's gone.
- **Default LLM changed** from `tiiuae/falcon-rw-1b` (a raw, non-instruction-
  tuned causal LM, prone to repetition/rambling for QA) to
  `Qwen/Qwen2.5-0.5B-Instruct` — smaller and instruction-tuned, which
  measurably helps an extractive-summarization prompt like this one. Still
  configurable back to Falcon or any other HF causal LM via one env var.
- **Anti-hallucination guardrail**: `RAGPipeline.ask()` checks retrieved
  context length before calling the LLM at all; below
  `CREDITRUST_MIN_CONTEXT_CHARS` (default 20), it returns a fixed
  "not enough information" answer instead of prompting the LLM with an
  empty/near-empty context (a common source of confident-sounding
  hallucination). Covered by `tests/test_rag_pipeline.py`.
- **Retrieval evaluation harness** (`scripts/evaluate_retrieval.py`) — a
  labeled query set + hit-rate@k / latency report, so a retrieval
  regression (bad embedding model swap, bad chunk size change) shows up as
  a number instead of a vibe. Documented as a proxy metric, not a
  substitute for human-graded relevance judgments (see README §13).
- **Cleaner generation**: switched to `return_full_text=False` on the HF
  `text-generation` pipeline so the model's own prompt echo is never part
  of the raw output, instead of relying entirely on string-splitting on
  `"Answer:"` after the fact (kept as a defensive fallback in
  `postprocess_answer`).

## 6. AI / prompt engineering improvements

- Reworded the prompt to explicitly instruct the model to say when it
  doesn't have enough information, rather than only asking it to "avoid
  making things up" (`rag/prompts.py`).
- Centralized the prompt template in one module instead of duplicated
  inline strings, so future prompt iteration happens in one place and is
  unit-tested (`tests/test_prompts.py`).

## 7. API layer (new)

There was no programmatic access to the pipeline before — only the
Streamlit UI. Added `src/creditrust/api/main.py` (FastAPI):

- `POST /api/v1/query` — typed request/response via Pydantic models
  (question length bounds, optional `top_k` override), structured error
  responses (422 on bad input, 503 if the vector index isn't built yet, 500
  with a logged traceback + generic message on unexpected failures — no
  internal error strings leaked to the client).
- `GET /health` — reports `vector_store_ready` and the active `llm_provider`
  for orchestrator readiness gating.
- Optional `X-API-Key` auth (`CREDITRUST_API_KEY`), CORS configuration, and
  a minimal in-memory sliding-window rate limiter (30 req/min/IP) on
  `/api/*` — enough to protect a single-instance internal tool without
  pulling in a new dependency or external store.
- Auto-generated OpenAPI docs at `/docs`.

## 8. Testing (new — there were zero tests before)

23 unit tests (`tests/`), all runnable in under a second with
`CREDITRUST_LLM_PROVIDER=mock` and **no model downloads**:

- `test_preprocessing.py` — product filtering (incl. the `all`/unknown-mode
  branches), empty/duplicate-narrative removal, text cleaning, short-
  narrative filtering, and the full `preprocess_complaints` pipeline.
- `test_validation.py` — schema detection, quality-report duplicate/null
  flags, hard-fail behavior on missing columns / zero rows.
- `test_chunking.py` — chunk splitting + source-index tracking, numpy
  scalar coercion and unsupported-type dropping in metadata sanitization.
- `test_config.py` — default paths anchor to `PROJECT_ROOT` regardless of
  CWD, env-var overrides apply, settings caching behaves correctly.
- `test_prompts.py` — prompt construction and answer post-processing.
- `test_rag_pipeline.py` — end-to-end `RAGPipeline.ask()` against a fake
  retriever + `MockLLMProvider`: normal answer path, the insufficient-
  context guardrail, and the empty-question rejection.
- `test_api.py` — FastAPI `TestClient` coverage of `/health`, the 503-when-
  index-missing path, a successful query via dependency override, 422 on an
  empty question, and API-key enforcement (401 without, 200 with).

CI (`.github/workflows/ci.yml`) now actually does something: `ruff check`,
`black --check`, then `pytest --cov=creditrust -m "not integration"` with
the mock LLM provider — previously it only ran `python --version`.

## 9. Security

- **XSS in the Streamlit UI**: the original `app.py` interpolated the raw
  LLM answer and raw complaint narrative excerpts directly into
  `st.markdown(..., unsafe_allow_html=True)` with no escaping. Since
  narrative text comes from consumer-submitted complaints (uncontrolled
  input) and the answer is LLM-generated from that same text, this was a
  live stored/reflected-XSS-shaped risk. Fixed by `html.escape()`-ing both
  the answer and every source excerpt before interpolation.
- **No secrets in code** (confirmed — none existed) and no secrets were
  introduced; added `.env.example` documenting every configurable value so
  a real `.env` (gitignored) is the only place secrets like
  `CREDITRUST_API_KEY` or `CREDITRUST_HF_TOKEN` would ever live.
- Fixed the `.gitignore` corruption (see P0 #3) that would have let a large
  binary index and raw consumer-complaint data get committed.
- API input validation via Pydantic (length-bounded `question` field) closes
  off unbounded-prompt-length abuse of the LLM endpoint.
- Docker image runs as a non-root user (`appuser`, uid 1000).

## 10. Performance / scalability notes

- Batch-inserts into Chroma (existing behavior, kept) rather than one
  `add_texts` call per row.
- `@st.cache_resource` on pipeline construction in the Streamlit app so the
  embedding model and LLM are loaded once per process, not once per
  question submission (the original app re-imported a module-level
  pipeline object each run, which worked but gave no control over reload
  behavior and coupled UI code to model-loading side effects at import
  time).
- Config-driven `index_batch_size` and `chunk_size`/`chunk_overlap` so
  indexing throughput/memory can be tuned without code changes as data
  volume grows.
- **Deliberately not done**: distributed processing (Spark/Ray), a
  managed vector DB, or async batched inference. At the dataset sizes this
  system targets (tens of thousands to low millions of complaints, single
  node), those add operational complexity without a corresponding
  performance need. If complaint volume grows 100–1000x, the first real
  bottleneck would be embedding throughput at index time — the natural next
  step there is batched GPU embedding, not a rewrite of this architecture.

## 11. Deployment (new)

- `Dockerfile` — non-root, healthchecked, `PYTHONPATH` set so
  `creditrust.*` imports resolve without a `pip install -e .`.
- `docker-compose.yml` — `api` (FastAPI/uvicorn) and `ui` (Streamlit)
  services sharing `data/`, `vector_store/`, and `logs/` bind mounts, so the
  index is built once and used by both.
- CI as described in §8.

## 12. Documentation

- `README.md` — fully rewritten: architecture diagram, package layout,
  technology choices with rationale, install/run/test/deploy instructions,
  configuration reference, example workflow, limitations, future work.
- `data/README.md` — data provenance + regeneration instructions (did not
  exist before).
- This file (`UPGRADE.md`).
- Inline docstrings on every module explaining *why*, not restating the
  code (module-level docstrings on `config.py`, `pipeline.py`, `llm.py`,
  `main.py`, `run_pipeline.py`, `evaluate_retrieval.py`).

## 13. What was deliberately not done, and why

Per the "do not overengineer" directive, the following were considered and
rejected as not justified by this project's actual scale/requirements:

- **Airflow/Dagster/Prefect** — the pipeline is two sequential batch stages
  run on demand; a DAG orchestrator adds infrastructure with no scheduling
  or complex-dependency need it would actually solve here.
- **Kafka/streaming ingestion** — CFPB complaint exports are periodic batch
  CSV dumps, not a live event stream.
- **Spark** — dataset sizes here (up to ~9.6M raw rows, ~1.3M after the
  expanded filter) run comfortably in pandas on a single node; distributed
  compute would add operational cost without solving an actual bottleneck.
- **A managed/hosted vector database** — Chroma's embedded, file-persisted
  mode is sufficient for a single-node deployment and avoids standing up
  and operating another service.
- **Kubernetes / multi-service microsplit beyond API+UI** — two Docker
  Compose services is the right granularity for this workload; a full
  microservices architecture would fragment a codebase this size for no
  operational benefit.
- **MLflow / a full experiment-tracking stack** — there is no hyperparameter
  search or model training in this project (the LLM and embedding model are
  both used off-the-shelf); a lightweight JSONL run log
  (`logs/pipeline_runs.jsonl`) covers the actual observability need without
  standing up a tracking server.

## 14. Remaining / open items

Documented in `README.md` §13 ("Limitations") and §14 ("Future
improvements"): no human-labeled relevance eval set, no production
drift/quality monitoring, static single API key rather than real
multi-user auth, and no incremental (only full-rebuild) indexing. None of
these block using the system as an internal analyst tool; they're the
right next investments if usage grows.

## 15. Final assessment

**Portfolio-grade.** The system now demonstrates, with working code and
passing tests rather than just a description: a validated/idempotent/
observable data pipeline; a testable ML/AI component built behind clean
interfaces (dependency-injected retriever + LLM provider) rather than
tightly coupled to one model or one UI; a properly designed, authenticated,
rate-limited API; a real test suite exercising data, ML-pipeline, and API
layers; a fixed security issue (XSS) and a fixed version-control hygiene
issue (corrupted `.gitignore` that would have let a multi-GB index and raw
data get committed); and deployment via Docker + a CI pipeline that
actually lints and tests instead of just checking out the repo. The
original prototype could not start (`app.py`'s import was broken) — this is
the single clearest before/after signal of the jump from "prototype" to
"production-ready."
