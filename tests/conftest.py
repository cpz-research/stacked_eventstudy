"""Shared test fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_panel(
    treatment_ages: tuple[int, ...],
    ages: tuple[int, ...],
    treatment_effect: float = 0.0,
    baseline_level: float = 10.0,
) -> pd.DataFrame:
    """Create a simple balanced panel for synthetic tests."""
    rows: list[dict[str, int | float]] = []
    unit_id = 1
    for treatment_age in treatment_ages:
        for _replicate in range(2):
            for age in ages:
                outcome = baseline_level + 0.5 * age
                if age >= treatment_age:
                    outcome += treatment_effect
                rows.append(
                    {
                        "id": unit_id,
                        "age": age,
                        "treatment_age": treatment_age,
                        "calendar_year": 2000 + age,
                        "outcome": float(outcome),
                    },
                )
            unit_id += 1
    return pd.DataFrame(rows)


def make_heterogeneous_panel(
    treatment_effects: dict[int, float],
    ages: tuple[int, ...],
    baseline_level: float = 10.0,
) -> pd.DataFrame:
    """Create a balanced panel with cohort-specific treatment effects."""
    rows: list[dict[str, int | float]] = []
    unit_id = 1
    for treatment_age, treatment_effect in treatment_effects.items():
        for _replicate in range(2):
            for age in ages:
                outcome = baseline_level + 0.5 * age
                if age >= treatment_age:
                    outcome += treatment_effect
                rows.append(
                    {
                        "id": unit_id,
                        "age": age,
                        "treatment_age": treatment_age,
                        "calendar_year": 2000 + age,
                        "outcome": float(outcome),
                    },
                )
            unit_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def minimal_panel() -> pd.DataFrame:
    """Return a small balanced panel with staggered treatment ages."""
    return make_panel(
        treatment_ages=(25, 26, 27),
        ages=(22, 23, 24, 25, 26, 27),
        treatment_effect=0.0,
        baseline_level=0.0,
    )


@pytest.fixture
def two_cohort_panel() -> pd.DataFrame:
    """Return a panel with at least two admissible treated cohorts."""
    return make_panel(
        treatment_ages=(25, 26, 27, 28),
        ages=(22, 23, 24, 25, 26, 27, 28, 29),
        treatment_effect=0.25,
        baseline_level=5.0,
    )


@pytest.fixture
def zero_effect_panel() -> pd.DataFrame:
    """Return a panel with no treatment effect."""
    return make_panel(
        treatment_ages=(25, 26, 27, 28),
        ages=(22, 23, 24, 25, 26, 27, 28, 29),
        treatment_effect=0.0,
        baseline_level=5.0,
    )


@pytest.fixture
def heterogeneous_effect_panel() -> pd.DataFrame:
    """Return a panel with deterministic cohort-specific treatment effects."""
    return make_heterogeneous_panel(
        treatment_effects={25: 1.0, 26: 2.0, 27: 3.0, 28: 4.0},
        ages=(22, 23, 24, 25, 26, 27, 28, 29),
        baseline_level=5.0,
    )
