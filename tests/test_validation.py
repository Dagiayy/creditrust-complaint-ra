from __future__ import annotations

import pandas as pd
import pytest

from creditrust.data.validation import (
    DataValidationError,
    profile_data_quality,
    validate_or_raise,
    validate_schema,
)


def test_validate_schema_detects_missing_columns():
    df = pd.DataFrame({"Product": ["a"]})
    missing = validate_schema(df)
    assert "Consumer complaint narrative" in missing


def test_validate_schema_passes_with_required_columns(sample_complaints_df):
    assert validate_schema(sample_complaints_df) == []


def test_profile_data_quality_flags_duplicates():
    df = pd.DataFrame(
        {
            "Product": ["Credit card"] * 4,
            "Consumer complaint narrative": ["same"] * 4,
        }
    )
    report = profile_data_quality(df)
    assert report.duplicate_narrative_pct == pytest.approx(75.0)
    assert any("duplicate" in w.lower() for w in report.warnings)


def test_profile_data_quality_empty_dataframe():
    df = pd.DataFrame({"Product": [], "Consumer complaint narrative": []})
    report = profile_data_quality(df)
    assert report.row_count == 0
    assert not report.is_valid


def test_validate_or_raise_raises_on_missing_columns():
    df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(DataValidationError):
        validate_or_raise(df)


def test_validate_or_raise_raises_on_zero_rows():
    df = pd.DataFrame({"Product": [], "Consumer complaint narrative": []})
    with pytest.raises(DataValidationError):
        validate_or_raise(df)


def test_validate_or_raise_passes_valid_data(sample_complaints_df):
    report = validate_or_raise(sample_complaints_df)
    assert report.is_valid
    assert report.row_count == len(sample_complaints_df)
