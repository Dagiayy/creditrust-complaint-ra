"""Text chunking for complaint narratives ahead of embedding/indexing."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from creditrust.config import get_settings


class Chunk(TypedDict):
    chunk: str
    source_index: int
    chunk_id: int


def chunk_texts(
    texts: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split each narrative in `texts` into overlapping chunks.

    `source_index` tracks the position of the originating row so chunk
    metadata (complaint id, product, etc.) can be joined back afterwards.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    all_chunks: list[Chunk] = []
    for idx, text in enumerate(texts):
        for chunk_id, chunk in enumerate(splitter.split_text(str(text))):
            all_chunks.append({"chunk": chunk, "source_index": idx, "chunk_id": chunk_id})
    return all_chunks


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata into the primitive types ChromaDB accepts."""
    sanitized: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            sanitized[key] = value.item()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
    return sanitized
