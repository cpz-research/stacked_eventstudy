"""Utility helpers."""

from collections.abc import Sequence
from math import sqrt

import pandas as pd


def coerce_covariates(covariates: Sequence[str]) -> tuple[str, ...]:
    """Return covariates as an immutable tuple."""
    return tuple(covariates)


def check_missing_columns(
    data: pd.DataFrame, columns: Sequence[str]
) -> tuple[str, ...]:
    """Return missing columns in the given data frame."""
    return tuple(column for column in columns if column not in data.columns)


def make_confidence_interval(
    estimate: float,
    std_error: float,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """Create a symmetric confidence interval."""
    half_width = z_value * std_error
    return estimate - half_width, estimate + half_width


def standard_error_from_variance(variance: float) -> float:
    """Return the standard error implied by a variance."""
    if variance < 0:
        return 0.0
    return sqrt(variance)
