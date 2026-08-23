from __future__ import annotations

import numpy as np

from creditrust.embeddings.chunking import chunk_texts, sanitize_metadata


def test_chunk_texts_splits_long_text_and_tracks_source_index():
    texts = ["short text", "x" * 1200]
    chunks = chunk_texts(texts, chunk_size=500, chunk_overlap=50)

    assert chunks[0]["source_index"] == 0
    long_text_chunks = [c for c in chunks if c["source_index"] == 1]
    assert len(long_text_chunks) >= 3
    assert [c["chunk_id"] for c in long_text_chunks] == list(range(len(long_text_chunks)))


def test_chunk_texts_empty_list_returns_empty():
    assert chunk_texts([]) == []


def test_sanitize_metadata_coerces_numpy_scalars():
    meta = {
        "complaint_id": np.int64(42),
        "score": np.float64(0.5),
        "flag": np.bool_(True),
        "product": "Credit card",
        "note": None,
    }
    sanitized = sanitize_metadata(meta)
    assert sanitized == {
        "complaint_id": 42,
        "score": 0.5,
        "flag": True,
        "product": "Credit card",
        "note": None,
    }
    assert isinstance(sanitized["complaint_id"], int)


def test_sanitize_metadata_drops_unsupported_types():
    meta = {"good": "keep", "bad": {"nested": "dict"}}
    sanitized = sanitize_metadata(meta)
    assert sanitized == {"good": "keep"}
