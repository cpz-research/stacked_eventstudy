"""Shared test fixtures."""

import pandas as pd
import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip Stata integration tests unless explicitly selected."""
    if "stata" in config.option.markexpr:
        return
    skip_stata = pytest.mark.skip(reason="run with `pytest -m stata`")
    for item in items:
        if "stata" in item.keywords:
            item.add_marker(skip_stata)


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


def make_grouped_panel(
    group_effects: dict[str, float],
    cohort_replicates: dict[str, dict[int, int]],
    ages: tuple[int, ...],
    cohort_effects: dict[int, float] | None = None,
    baseline_level: float = 10.0,
) -> pd.DataFrame:
    """Create a balanced panel with group-specific treatment effects."""
    rows: list[dict[str, int | float | str]] = []
    unit_id = 1
    cohort_effects = {} if cohort_effects is None else cohort_effects
    for group_value, treatment_effect in group_effects.items():
        for treatment_age, replicates in cohort_replicates[group_value].items():
            for _replicate in range(replicates):
                for age in ages:
                    outcome = baseline_level + 0.5 * age
                    if age >= treatment_age:
                        outcome += treatment_effect + cohort_effects.get(
                            treatment_age,
                            0.0,
                        )
                    rows.append(
                        {
                            "id": unit_id,
                            "age": age,
                            "treatment_age": treatment_age,
                            "calendar_year": 2000 + age,
                            "outcome": float(outcome),
                            "segment": group_value,
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


@pytest.fixture
def grouped_effect_panel() -> pd.DataFrame:
    """Return a panel with deterministic group-specific treatment effects."""
    return make_grouped_panel(
        group_effects={"a": 1.0, "b": 3.0, "c": 5.0},
        cohort_replicates={
            "a": {25: 2, 26: 2, 27: 2, 28: 2},
            "b": {25: 2, 26: 2, 27: 2, 28: 2},
            "c": {25: 2, 26: 2, 27: 2, 28: 2},
        },
        ages=(22, 23, 24, 25, 26, 27, 28, 29),
        baseline_level=5.0,
    )


@pytest.fixture
def uneven_grouped_effect_panel() -> pd.DataFrame:
    """Return a grouped panel with uneven cohort composition."""
    return make_grouped_panel(
        group_effects={"a": 1.0, "b": 3.0},
        cohort_replicates={
            "a": {25: 4, 26: 1, 27: 2, 28: 2},
            "b": {25: 1, 26: 4, 27: 2, 28: 2},
        },
        ages=(22, 23, 24, 25, 26, 27, 28, 29),
        cohort_effects={25: 0.0, 26: 2.0, 27: 0.0, 28: 0.0},
        baseline_level=5.0,
    )
