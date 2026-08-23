from __future__ import annotations

import pytest

from creditrust.data.preprocessing import (
    clean_text,
    filter_products,
    preprocess_complaints,
    remove_duplicate_narratives,
    remove_empty_narratives,
    remove_short_narratives,
)


def test_filter_products_strict_excludes_mortgage(sample_complaints_df):
    result = filter_products(sample_complaints_df, mode="strict")
    assert "Mortgage" not in result["Product"].values
    assert set(result["Product"].unique()) <= {"Credit card", "Personal loan", "Savings account"}


def test_filter_products_all_mode_returns_everything(sample_complaints_df):
    result = filter_products(sample_complaints_df, mode="all")
    assert len(result) == len(sample_complaints_df)


def test_filter_products_unknown_mode_raises(sample_complaints_df):
    with pytest.raises(ValueError):
        filter_products(sample_complaints_df, mode="bogus")


def test_remove_empty_narratives_drops_nulls(sample_complaints_df):
    result = remove_empty_narratives(sample_complaints_df)
    assert result["Consumer complaint narrative"].notna().all()
    assert len(result) == 5


def test_remove_duplicate_narratives_keeps_first_only():
    import pandas as pd

    df = pd.DataFrame({"Consumer complaint narrative": ["same text", "same text", "different"]})
    result = remove_duplicate_narratives(df)
    assert len(result) == 2


def test_clean_text_strips_boilerplate_and_lowercases():
    raw = "I am writing to file a complaint about my ACCOUNT!!"
    cleaned = clean_text(raw)
    assert "i am writing to file a complaint" not in cleaned
    assert cleaned == cleaned.lower()
    assert "account" in cleaned


def test_remove_short_narratives_filters_by_word_count():
    import pandas as pd

    df = pd.DataFrame({"cleaned_narrative": ["one two", "one two three four five six"]})
    result = remove_short_narratives(df, min_words=3)
    assert len(result) == 1
    assert result.iloc[0]["cleaned_narrative"] == "one two three four five six"


def test_preprocess_complaints_end_to_end(sample_complaints_df):
    result = preprocess_complaints(sample_complaints_df, filter_mode="strict")
    # Mortgage row and the null-narrative row are gone, and the exact-duplicate
    # "Credit card" narrative is deduplicated down to one occurrence.
    assert "Mortgage" not in result["Product"].values
    assert result["Consumer complaint narrative"].notna().all()
    assert result["Consumer complaint narrative"].duplicated().sum() == 0
    assert "cleaned_narrative" in result.columns


def test_preprocess_complaints_applies_min_word_filter(sample_complaints_df):
    result = preprocess_complaints(
        sample_complaints_df, filter_mode="strict", apply_min_word_filter=True, min_words=3
    )
    assert "too short" not in result["Consumer complaint narrative"].values
