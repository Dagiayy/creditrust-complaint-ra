"""Data cleaning and filtering for raw CFPB consumer complaint exports.

Pipeline stage: Extract (caller) -> Validate -> Transform (this module).
"""

from __future__ import annotations

import re

import pandas as pd

from creditrust.logging_config import get_logger

logger = get_logger(__name__)

STRICT_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers",
]

EXPANDED_PRODUCTS = STRICT_PRODUCTS + [
    "Checking or savings account",
    "Credit card or prepaid card",
    "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Money transfer, virtual currency, or money service",
]

_BOILERPLATE_PATTERN = re.compile(r"\b(i am writing to file a complaint|i would like to report)\b")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s.,!?']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def load_data(path: str) -> pd.DataFrame:
    """Load the complaint dataset from CSV."""
    logger.info("Loading raw complaint data from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def filter_products(df: pd.DataFrame, mode: str = "strict") -> pd.DataFrame:
    """Filter rows by product category.

    - strict: only the 5 core products the business cares about.
    - expanded: strict + adjacent product categories with narrative volume.
    - all: no product filtering (still requires the Product column to exist).
    """
    if mode == "strict":
        products = STRICT_PRODUCTS
    elif mode == "expanded":
        products = EXPANDED_PRODUCTS
    elif mode == "all":
        return df
    else:
        raise ValueError(f"Unknown filter mode: {mode!r}. Expected 'strict', 'expanded', or 'all'.")

    filtered = df[df["Product"].isin(products)]
    logger.info("Product filter %r: %d -> %d rows", mode, len(df), len(filtered))
    return filtered


def remove_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without a complaint narrative."""
    filtered = df[df["Consumer complaint narrative"].notna()]
    logger.info("Removed empty narratives: %d -> %d rows", len(df), len(filtered))
    return filtered


def remove_duplicate_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose narrative text is an exact duplicate of an earlier row.

    Exact-duplicate narratives (common with bulk/templated CFPB submissions)
    inflate retrieval results with redundant chunks without adding signal.
    """
    before = len(df)
    deduped = df.drop_duplicates(subset=["Consumer complaint narrative"], keep="first")
    logger.info("Removed duplicate narratives: %d -> %d rows", before, len(deduped))
    return deduped


def clean_text(text: str) -> str:
    """Normalize a single narrative: lowercase, strip boilerplate/punctuation noise."""
    text = str(text).lower()
    text = _BOILERPLATE_PATTERN.sub("", text)
    text = _NON_ALNUM_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to all complaint narratives, producing `cleaned_narrative`."""
    df = df.copy()
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)
    return df


def remove_short_narratives(df: pd.DataFrame, min_words: int = 10) -> pd.DataFrame:
    """Remove complaints whose cleaned narrative is shorter than `min_words`."""
    df = df.copy()
    word_count = df["cleaned_narrative"].str.split().str.len()
    filtered = df[word_count >= min_words]
    logger.info("Removed narratives shorter than %d words: %d -> %d rows", min_words, len(df), len(filtered))
    return filtered


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Persist preprocessed data to CSV."""
    df.to_csv(output_path, index=False)
    logger.info("Saved %d cleaned rows to %s", len(df), output_path)


def preprocess_complaints(
    df: pd.DataFrame,
    apply_min_word_filter: bool = False,
    min_words: int = 5,
    filter_mode: str = "strict",
    drop_duplicate_narratives: bool = True,
) -> pd.DataFrame:
    """Full preprocessing pipeline: filter -> dedupe -> clean -> (optional) length filter.

    Order matters: product filtering and empty-narrative removal happen before
    the (relatively expensive) text cleaning step so cleaning only runs on
    rows that will actually be kept.
    """
    df = filter_products(df, mode=filter_mode)
    df = remove_empty_narratives(df)
    if drop_duplicate_narratives:
        df = remove_duplicate_narratives(df)
    df = clean_narratives(df)

    if apply_min_word_filter:
        df = remove_short_narratives(df, min_words)

    return df
