"""Retriever construction on top of the persisted Chroma vector store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from creditrust.config import get_settings
from creditrust.embeddings.indexer import get_embedding_function, load_vector_store

if TYPE_CHECKING:
    from langchain_core.documents import Document


class Retriever(Protocol):
    def invoke(self, query: str) -> list[Document]: ...


def vector_store_exists(persist_directory: Path | str | None = None) -> bool:
    settings = get_settings()
    path = Path(persist_directory or settings.vector_store_dir)
    return path.exists() and any(path.iterdir()) if path.exists() else False


def get_retriever(
    persist_directory: Path | str | None = None,
    top_k: int | None = None,
) -> Retriever:
    """Build a retriever over the persisted vector store.

    Raises FileNotFoundError with an actionable message if the index has not
    been built yet, instead of failing deep inside Chroma with an opaque error.
    """
    settings = get_settings()
    persist_directory = Path(persist_directory or settings.vector_store_dir)

    if not vector_store_exists(persist_directory):
        raise FileNotFoundError(
            f"No vector index found at {persist_directory}. "
            "Run `python scripts/run_pipeline.py index` first."
        )

    vectorstore = load_vector_store(persist_directory, get_embedding_function())
    return vectorstore.as_retriever(search_kwargs={"k": top_k or settings.top_k})
