"""Lightweight data-quality validation for the raw CFPB complaint export.

No heavy schema-validation dependency (e.g. pandera/great_expectations) is
introduced here deliberately — the checks are simple and dependency-light,
which keeps the pipeline installable in constrained environments while still
catching the failure modes that actually matter for this dataset: missing
required columns, an empty frame, and a narrative column that is entirely
null (the single most common way this pipeline silently produces zero
usable rows downstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ["Product", "Consumer complaint narrative"]


class DataValidationError(ValueError):
    """Raised when the input dataset fails a hard validation check."""


@dataclass
class DataQualityReport:
    row_count: int
    column_count: int
    missing_required_columns: list[str] = field(default_factory=list)
    null_narrative_pct: float = 0.0
    duplicate_row_pct: float = 0.0
    duplicate_narrative_pct: float = 0.0
    product_value_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_columns and self.row_count > 0

    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_required_columns": self.missing_required_columns,
            "null_narrative_pct": round(self.null_narrative_pct, 4),
            "duplicate_row_pct": round(self.duplicate_row_pct, 4),
            "duplicate_narrative_pct": round(self.duplicate_narrative_pct, 4),
            "product_value_counts": self.product_value_counts,
            "warnings": self.warnings,
            "is_valid": self.is_valid,
        }


def validate_schema(df: pd.DataFrame, required_columns: list[str] | None = None) -> list[str]:
    """Return the list of required columns missing from ``df``."""
    required = required_columns or REQUIRED_COLUMNS
    return [col for col in required if col not in df.columns]


def profile_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Compute a data-quality report used for pipeline observability and gating."""
    missing_cols = validate_schema(df)
    report = DataQualityReport(
        row_count=len(df),
        column_count=len(df.columns),
        missing_required_columns=missing_cols,
    )

    if missing_cols:
        report.warnings.append(f"Missing required columns: {missing_cols}")
        return report

    if len(df) == 0:
        report.warnings.append("Dataframe has zero rows.")
        return report

    narrative_col = "Consumer complaint narrative"
    report.null_narrative_pct = float(df[narrative_col].isna().mean() * 100)
    report.duplicate_row_pct = float(df.duplicated().mean() * 100)
    report.duplicate_narrative_pct = (
        float(df[narrative_col].dropna().duplicated().mean() * 100)
        if df[narrative_col].notna().any()
        else 0.0
    )
    report.product_value_counts = {str(k): int(v) for k, v in df["Product"].value_counts().head(20).items()}

    if report.null_narrative_pct > 95:
        report.warnings.append(
            f"{report.null_narrative_pct:.1f}% of narratives are null — check the source export."
        )
    if report.duplicate_narrative_pct > 20:
        report.warnings.append(
            f"{report.duplicate_narrative_pct:.1f}% of non-null narratives are exact duplicates."
        )

    return report


def validate_or_raise(df: pd.DataFrame, required_columns: list[str] | None = None) -> DataQualityReport:
    """Validate ``df`` and raise :class:`DataValidationError` on hard failures."""
    report = profile_data_quality(df)
    if report.missing_required_columns:
        raise DataValidationError(
            f"Input data is missing required columns: {report.missing_required_columns}. "
            f"Found columns: {list(df.columns)}"
        )
    if report.row_count == 0:
        raise DataValidationError("Input data has zero rows after loading.")
    return report
