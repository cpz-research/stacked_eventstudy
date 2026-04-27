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
