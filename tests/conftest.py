"""Shared test fixtures."""

from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def minimal_panel() -> pd.DataFrame:
    """Return a small balanced panel with staggered treatment ages."""
    rows: list[dict[str, int | float]] = []
    for unit_id, treatment_age in [(1, 25), (2, 25), (3, 26), (4, 26), (5, 27), (6, 27)]:
        for age in range(22, 28):
            rows.append(
                {
                    "id": unit_id,
                    "age": age,
                    "treatment_age": treatment_age,
                    "calendar_year": 2000 + age,
                    "outcome": float(age - 20),
                },
            )
    return pd.DataFrame(rows)


@pytest.fixture
def two_cohort_panel() -> pd.DataFrame:
    """Return a panel with at least two admissible treated cohorts."""
    rows: list[dict[str, int | float]] = []
    for unit_id, treatment_age in [
        (1, 25),
        (2, 25),
        (3, 26),
        (4, 26),
        (5, 27),
        (6, 27),
        (7, 28),
        (8, 28),
    ]:
        for age in range(22, 30):
            rows.append(
                {
                    "id": unit_id,
                    "age": age,
                    "treatment_age": treatment_age,
                    "calendar_year": 2000 + age,
                    "outcome": float(age - 20 + 0.25 * (age >= treatment_age)),
                },
            )
    return pd.DataFrame(rows)
