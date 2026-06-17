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


def test_validate_rejects_missing_heterogeneity_column(
    minimal_panel: pd.DataFrame,
) -> None:
    """Reject a missing heterogeneity column."""
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
        heterogeneity_col="segment",
    )
    assert not result.is_valid
    assert any("segment" in error for error in result.errors)


def test_validate_rejects_heterogeneity_column_as_covariate(
    minimal_panel: pd.DataFrame,
) -> None:
    """Reject using the heterogeneity column as a covariate."""
    data = minimal_panel.copy()
    data["segment"] = "a"
    result = validate_stacked_eventstudy(
        data=data,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
        covariates=("segment",),
        heterogeneity_col="segment",
    )
    assert not result.is_valid
    assert any("heterogeneity_col must not" in error for error in result.errors)


def test_validate_rejects_missing_heterogeneity_value(
    minimal_panel: pd.DataFrame,
) -> None:
    """Reject missing values in the heterogeneity column."""
    invalid = minimal_panel.copy()
    invalid["segment"] = "a"
    invalid.loc[0, "segment"] = pd.NA
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
        heterogeneity_col="segment",
    )
    assert not result.is_valid
    assert any("heterogeneity_value (1 missing)" in error for error in result.errors)


def test_validate_rejects_time_varying_heterogeneity_value(
    minimal_panel: pd.DataFrame,
) -> None:
    """Reject heterogeneity values that vary within individual."""
    invalid = minimal_panel.copy()
    invalid["segment"] = "a"
    first_unit = invalid["id"].min()
    invalid.loc[
        (invalid["id"] == first_unit) & (invalid["age"] == invalid["age"].max()),
        "segment",
    ] = "b"
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
        heterogeneity_col="segment",
    )
    assert not result.is_valid
    assert any("heterogeneity_col must be constant" in error for error in result.errors)


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


def test_validate_requires_complete_control_event_window(
    minimal_panel: pd.DataFrame,
) -> None:
    """Require complete control support in the requested event-time window."""
    missing_aligned_control_age = 24
    incomplete_controls = minimal_panel.loc[
        ~(
            (minimal_panel["treatment_age"].isin([26, 27]))
            & (minimal_panel["age"] == missing_aligned_control_age)
        ),
    ]
    result = validate_stacked_eventstudy(
        data=incomplete_controls,
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
    assert "missing_control_event_times" in set(
        result.cohort_diagnostics["drop_reason"],
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


def test_estimate_rejects_unknown_covariance_policy(
    minimal_panel: pd.DataFrame,
) -> None:
    """Reject unsupported covariance policies."""
    with pytest.raises(
        ValueError,
        match="covariance_policy must be either 'small_sample_correction' or 'none'",
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
            covariance_policy="unknown",
        )


def test_estimate_rejects_unknown_heterogeneity_weighting(
    minimal_panel: pd.DataFrame,
) -> None:
    """Reject unsupported heterogeneity weighting schemes."""
    data = minimal_panel.copy()
    data["segment"] = "a"
    with pytest.raises(
        ValueError,
        match="heterogeneity_weighting must be either 'within' or 'overall'",
    ):
        estimate_stacked_eventstudy(
            data=data,
            id_col="id",
            age_col="age",
            treatment_age_col="treatment_age",
            outcome_col="outcome",
            l_min=-2,
            l_max=1,
            control_window=2,
            reference_event_time=-1,
            calendar_year_col="calendar_year",
            heterogeneity_col="segment",
            heterogeneity_weighting="pooled",
        )
