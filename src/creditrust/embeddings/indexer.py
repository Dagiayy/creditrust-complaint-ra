"""Builds and loads the Chroma vector index of embedded complaint chunks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from tqdm import tqdm

from creditrust.config import get_settings
from creditrust.embeddings.chunking import chunk_texts, sanitize_metadata
from creditrust.logging_config import get_logger

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)


def get_embedding_function(model_name: str | None = None) -> HuggingFaceEmbeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=model_name or settings.embedding_model_name)


def load_vector_store(
    persist_directory: Path | str | None = None,
    embedding_function: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Load (without rebuilding) the persisted Chroma index."""
    from langchain_chroma import Chroma

    settings = get_settings()
    persist_directory = Path(persist_directory or settings.vector_store_dir)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_function or get_embedding_function(),
    )


def build_vector_store(
    df: pd.DataFrame,
    persist_directory: Path | str | None = None,
    text_column: str = "cleaned_narrative",
    embedding_function: HuggingFaceEmbeddings | None = None,
    batch_size: int | None = None,
) -> int:
    """Chunk `df[text_column]`, embed each chunk, and persist to a Chroma index.

    Returns the number of chunks indexed. Metadata carries `complaint_id`,
    `product`, and `chunk_id` so answers can cite their source complaint.
    """
    settings = get_settings()
    persist_directory = Path(persist_directory or settings.vector_store_dir)
    persist_directory.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or settings.index_batch_size

    logger.info("Chunking %d narratives...", len(df))
    chunks = chunk_texts(df[text_column].tolist())
    logger.info("Created %d chunks from %d narratives", len(chunks), len(df))

    if not chunks:
        logger.warning("No chunks produced — vector store will not be updated.")
        return 0

    vectorstore = load_vector_store(persist_directory, embedding_function)

    texts = [c["chunk"] for c in chunks]
    metadatas = [
        sanitize_metadata(
            {
                "complaint_id": str(df.iloc[c["source_index"]].get("Complaint ID", c["source_index"])),
                "product": df.iloc[c["source_index"]].get("Product", "Unknown"),
                "chunk_id": c["chunk_id"],
            }
        )
        for c in chunks
    ]

    logger.info("Embedding and inserting %d chunks in batches of %d...", len(texts), batch_size)
    for i in tqdm(range(0, len(texts), batch_size), desc="Indexing"):
        vectorstore.add_texts(texts[i : i + batch_size], metadatas=metadatas[i : i + batch_size])

    logger.info("Indexed %d chunks into %s", len(texts), persist_directory)
    return len(texts)
