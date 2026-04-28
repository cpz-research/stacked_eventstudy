"""Validation tests."""

import pandas as pd
import pytest

from stacked_eventstudy import estimate_stacked_eventstudy, validate_stacked_eventstudy


def test_validate_reports_valid_panel(minimal_panel: pd.DataFrame) -> None:
    """Validate a balanced minimal panel."""
    result = validate_stacked_eventstudy(
        data=minimal_panel,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
    )
    assert result.is_valid
    assert not result.errors
    assert result.admissible_cohort_range == (25, 25)


def test_validate_rejects_duplicate_id_age(minimal_panel: pd.DataFrame) -> None:
    """Reject duplicated id-age observations."""
    duplicated = pd.concat([minimal_panel, minimal_panel.iloc[[0]]], ignore_index=True)
    result = validate_stacked_eventstudy(
        data=duplicated,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
    )
    assert not result.is_valid
    assert any("Duplicated id-age" in error for error in result.errors)


def test_validate_rejects_missing_treatment_age(minimal_panel: pd.DataFrame) -> None:
    """Reject missing treatment age."""
    invalid = minimal_panel.copy()
    invalid.loc[0, "treatment_age"] = pd.NA
    result = validate_stacked_eventstudy(
        data=invalid,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
    )
    assert not result.is_valid
    assert any("treatment_age (1 missing)" in error for error in result.errors)


def test_validate_rejects_missing_covariate(minimal_panel: pd.DataFrame) -> None:
    """Reject missing formula covariates before estimation."""
    invalid = minimal_panel.copy()
    invalid["covariate"] = invalid["age"]
    invalid.loc[0, "covariate"] = pd.NA
    result = validate_stacked_eventstudy(
        data=invalid,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
        covariates=("covariate",),
    )
    assert not result.is_valid
    assert any("covariate (1 missing)" in error for error in result.errors)


def test_validate_rejects_infeasible_window(minimal_panel: pd.DataFrame) -> None:
    """Reject an infeasible post-birth horizon."""
    result = validate_stacked_eventstudy(
        data=minimal_panel,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=3,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
    )
    assert not result.is_valid
    assert any(
        error == "l_max must be less than or equal to control_window - 1."
        for error in result.errors
    )


def test_estimate_rejects_unknown_backend(minimal_panel: pd.DataFrame) -> None:
    """Reject unsupported regression backends."""
    with pytest.raises(
        ValueError,
        match="backend must be either 'statsmodels' or 'pyfixest'",
    ):
        estimate_stacked_eventstudy(
            data=minimal_panel,
            id_col="id",
            age_col="age",
            treatment_age_col="treatment_age",
            outcome_col="outcome",
            l_min=-2,
            l_max=1,
            control_window=2,
            reference_event_time=-1,
            calendar_year_col="calendar_year",
            backend="unknown",
        )
