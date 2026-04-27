"""End-to-end estimator smoke tests."""

import pandas as pd

from stacked_eventstudy import estimate_stacked_eventstudy


def test_estimate_stacked_eventstudy_runs(minimal_panel: pd.DataFrame) -> None:
    """Run the stacked estimator on a small synthetic panel."""
    result = estimate_stacked_eventstudy(
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
        return_stacked_data=True,
    )
    assert not result.cohort_params.empty
    assert not result.average_params.empty
    assert not result.cohort_weights.empty
    assert result.stacked_data is not None
    assert -1 not in set(result.cohort_params["event_time"])


def test_estimate_stacked_eventstudy_returns_square_vcov(
    two_cohort_panel: pd.DataFrame,
) -> None:
    """Return a full aggregated covariance matrix across event times."""
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
    event_times = result.average_params["event_time"].tolist()
    assert list(result.vcov_average.index) == event_times
    assert list(result.vcov_average.columns) == event_times
    assert result.vcov_average.shape[0] == result.vcov_average.shape[1]
