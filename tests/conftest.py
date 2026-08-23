from __future__ import annotations

import pandas as pd
import pytest

from creditrust.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test sees a fresh Settings instance reflecting any env changes."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_complaints_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Complaint ID": [1, 2, 3, 4, 5, 6],
            "Product": [
                "Credit card",
                "Personal loan",
                "Mortgage",
                "Credit card",
                "Savings account",
                "Credit card",
            ],
            "Consumer complaint narrative": [
                "I am writing to file a complaint about unauthorized charges on my account that I never made.",
                "My personal loan payments were misapplied and I was charged extra fees repeatedly for months.",
                None,
                "I am writing to file a complaint about unauthorized charges on my account that I never made.",
                "too short",
                "The savings account had a hold placed on it for over three weeks without any explanation given.",
            ],
        }
    )
