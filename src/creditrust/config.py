"""Centralized, environment-driven configuration for the CrediTrust RAG system.

All paths are resolved relative to the project root (the parent of ``src/``)
rather than the process's current working directory, so the pipeline behaves
identically whether it is launched as ``python scripts/run_pipeline.py``,
``streamlit run app.py``, ``uvicorn creditrust.api.main:app``, or from a test
runner in a different directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, overridable via environment variables or a `.env` file.

    Every setting can be overridden with an environment variable of the same
    (upper-cased) name, e.g. ``CREDITRUST_LLM_MODEL_NAME=google/flan-t5-base``.
    """

    model_config = SettingsConfigDict(
        env_prefix="CREDITRUST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Filesystem layout -------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "complaints.csv"
    filtered_data_path: Path = PROJECT_ROOT / "data" / "filtered_complaints.csv"
    vector_store_dir: Path = PROJECT_ROOT / "vector_store" / "chroma_index"
    logs_dir: Path = PROJECT_ROOT / "logs"
    pipeline_run_log: Path = PROJECT_ROOT / "logs" / "pipeline_runs.jsonl"

    # --- Data filtering / cleaning ------------------------------------------
    filter_mode: Literal["strict", "expanded", "all"] = "strict"
    min_narrative_words: int = 5
    apply_min_word_filter: bool = False

    # --- Chunking / embeddings ----------------------------------------------
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    index_batch_size: int = 5000

    # --- Retrieval -----------------------------------------------------------
    top_k: int = 5
    min_context_chars: int = 20
    """Below this amount of retrieved context, the pipeline refuses to answer
    rather than let the LLM hallucinate from an empty/near-empty context."""

    # --- LLM -------------------------------------------------------------------
    llm_provider: Literal["local", "mock"] = "local"
    llm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_max_new_tokens: int = 256
    llm_temperature: float = 0.3
    hf_token: str | None = None

    # --- API ---------------------------------------------------------------
    api_key: str | None = None
    """If set, the FastAPI service requires this value in the `X-API-Key` header."""
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Logging -------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator(
        "data_dir",
        "raw_data_path",
        "filtered_data_path",
        "vector_store_dir",
        "logs_dir",
        "pipeline_run_log",
        mode="before",
    )
    @classmethod
    def _resolve_relative_to_project_root(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()
