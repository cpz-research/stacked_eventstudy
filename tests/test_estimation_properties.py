"""Behavioral tests for estimation outputs."""

from math import isclose

import pandas as pd

from stacked_eventstudy import estimate_stacked_eventstudy


def test_zero_effect_panel_recovers_near_zero_effects(
    zero_effect_panel: pd.DataFrame,
) -> None:
    """Recover approximately zero dynamic effects when treatment has no effect."""
    result = estimate_stacked_eventstudy(
        data=zero_effect_panel,
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
    assert result.average_params["estimate"].abs().max() < 1e-8


def test_aggregation_matches_weighted_cohort_average(
    two_cohort_panel: pd.DataFrame,
) -> None:
    """Match the returned average effects to weighted cohort averages."""
    result = estimate_stacked_eventstudy(
        data=two_cohort_panel,
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
    assert isclose(result.cohort_weights["weight"].sum(), 1.0, rel_tol=0.0, abs_tol=1e-12)

    merged = result.cohort_params.merge(
        result.cohort_weights,
        on="subevent",
        how="left",
        validate="many_to_one",
    )
    manual_average = (
        merged.assign(weighted_estimate=merged["estimate"] * merged["weight"])
        .groupby("event_time", as_index=False)["weighted_estimate"]
        .sum()
        .rename(columns={"weighted_estimate": "manual_estimate"})
    )
    comparison = result.average_params.merge(manual_average, on="event_time", how="left")
    assert (comparison["estimate"] - comparison["manual_estimate"]).abs().max() < 1e-10


def test_pre_birth_scaling_matches_manual_scaling(
    two_cohort_panel: pd.DataFrame,
) -> None:
    """Match returned scaled averages to cohort-level manual scaling."""
    unscaled_result = estimate_stacked_eventstudy(
        data=two_cohort_panel,
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
    scaled_result = estimate_stacked_eventstudy(
        data=two_cohort_panel,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
        scale="pre_birth",
    )

    pre_birth_levels = (
        two_cohort_panel.loc[
            two_cohort_panel["age"] == two_cohort_panel["treatment_age"] - 1,
            ["treatment_age", "outcome"],
        ]
        .groupby("treatment_age", as_index=False)["outcome"]
        .mean()
        .rename(columns={"treatment_age": "subevent", "outcome": "pre_birth_level"})
    )
    manual_scaled = (
        unscaled_result.cohort_params.merge(
            pre_birth_levels,
            on="subevent",
            how="left",
            validate="many_to_one",
        )
        .merge(
            unscaled_result.cohort_weights,
            on="subevent",
            how="left",
            validate="many_to_one",
        )
    )
    manual_scaled["scaled_estimate"] = (
        manual_scaled["estimate"] / manual_scaled["pre_birth_level"]
    )
    manual_average = (
        manual_scaled.assign(weighted_estimate=manual_scaled["scaled_estimate"] * manual_scaled["weight"])
        .groupby("event_time", as_index=False)["weighted_estimate"]
        .sum()
        .rename(columns={"weighted_estimate": "manual_estimate"})
    )
    comparison = scaled_result.average_params.merge(manual_average, on="event_time", how="left")
    assert set(scaled_result.average_params["scale"]) == {"pre_birth"}
    assert (comparison["estimate"] - comparison["manual_estimate"]).abs().max() < 1e-10


def test_heterogeneous_effect_panel_recovers_cohort_specific_effects(
    heterogeneous_effect_panel: pd.DataFrame,
) -> None:
    """Recover deterministic cohort-specific post-treatment effects."""
    result = estimate_stacked_eventstudy(
        data=heterogeneous_effect_panel,
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
    expected = {
        (25, -2): 0.0,
        (25, 0): 1.0,
        (25, 1): 1.0,
        (26, -2): 0.0,
        (26, 0): 2.0,
        (26, 1): 2.0,
    }
    for row in result.cohort_params.itertuples(index=False):
        target = expected[(row.subevent, row.event_time)]
        assert isclose(row.estimate, target, rel_tol=0.0, abs_tol=1e-10)

    expected_average = {-2: 0.0, 0: 1.5, 1: 1.5}
    for row in result.average_params.itertuples(index=False):
        assert isclose(
            row.estimate,
            expected_average[row.event_time],
            rel_tol=0.0,
            abs_tol=1e-10,
        )


def test_result_tables_expose_expected_columns(
    two_cohort_panel: pd.DataFrame,
) -> None:
    """Return result tables with the expected public columns."""
    result = estimate_stacked_eventstudy(
        data=two_cohort_panel,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
        return_stacked_data=True,
    )
    assert {
        "subevent",
        "event_time",
        "term_label",
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "scale",
        "n_treated_individuals",
        "n_control_individuals",
        "n_treated_obs",
        "n_control_obs",
    }.issubset(result.cohort_params.columns)
    assert {
        "event_time",
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "n_cohorts",
        "scale",
    }.issubset(result.average_params.columns)
    assert {"subevent", "n_individuals", "weight"} == set(result.cohort_weights.columns)
    assert result.stacked_data is not None
