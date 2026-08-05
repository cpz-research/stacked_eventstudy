"""Stack construction helpers."""

import pandas as pd

from stacked_eventstudy.types import EstimatorConfig, StackedEventStudyValidation


def build_stacked_data(
    data: pd.DataFrame,
    config: EstimatorConfig,
    validation: StackedEventStudyValidation,
) -> pd.DataFrame:
    """Build the full stacked event-study dataset."""
    if validation.admissible_cohort_range is None:
        msg = "No admissible cohorts are available for stack construction."
        raise ValueError(msg)

    admissible_cohorts = validation.cohort_diagnostics.loc[
        validation.cohort_diagnostics["admissible"],
        "subevent",
    ].tolist()
    stacks = [
        build_subevent_stack(data=data, config=config, subevent=int(subevent))
        for subevent in admissible_cohorts
    ]
    stacked = pd.concat(stacks, ignore_index=True)
    stacked["unit_subevent_id"] = (
        stacked["unit_id"].astype(str) + "__" + stacked["subevent"].astype(str)
    )
    return stacked.sort_values(["subevent", "unit_id", "age"]).reset_index(drop=True)


def build_subevent_stack(
    data: pd.DataFrame,
    config: EstimatorConfig,
    subevent: int,
) -> pd.DataFrame:
    """Build the stack for one treatment-age cohort."""
    treated = data.loc[data["treatment_age"] == subevent].copy()
    treated["subevent"] = subevent
    treated["event_time"] = treated["age"] - subevent
    treated = treated.loc[
        (treated["event_time"] >= config.l_min)
        & (treated["event_time"] <= config.l_max)
    ]
    if not config.allow_unbalanced_treated_panel:
        treated = retain_complete_treated_units(
            treated=treated,
            l_min=config.l_min,
            l_max=config.l_max,
        )
    treated["treated_in_subevent"] = 1

    controls = data.loc[
        data["treatment_age"].between(subevent + 1, subevent + config.control_window),
    ].copy()
    controls = controls.loc[controls["age"] < controls["treatment_age"]]
    controls["subevent"] = subevent
    controls["event_time"] = controls["age"] - subevent
    controls = controls.loc[
        (controls["event_time"] >= config.l_min)
        & (controls["event_time"] <= config.l_max)
    ]
    controls["treated_in_subevent"] = 0

    subevent_data = pd.concat([treated, controls], ignore_index=True)
    subevent_data["stack_weight"] = subevent_data["input_weight"]
    return subevent_data


def retain_complete_treated_units(
    treated: pd.DataFrame,
    l_min: int,
    l_max: int,
) -> pd.DataFrame:
    """Retain focal treated units observed at every requested event time."""
    if treated.empty:
        return treated
    required_event_times = frozenset(range(l_min, l_max + 1))
    observed_event_times = treated.groupby("unit_id")["event_time"].agg(frozenset)
    complete_unit_ids = observed_event_times.index[
        observed_event_times.eq(required_event_times)
    ]
    return treated.loc[treated["unit_id"].isin(complete_unit_ids)].copy()
