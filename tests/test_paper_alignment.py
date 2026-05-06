"""Tests that align the implementation with the paper's estimator."""

from math import isclose

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from stacked_eventstudy import estimate_stacked_eventstudy

STRICT_ATOL = 1e-10
RECOVERY_ATOL = 1e-8
UNBALANCED_COHORT = 25
COMPARISON_COHORT = 26
DROPPED_FOCAL_AGE = 23
UNBALANCED_COHORT_OBSERVATIONS = 7
COMPARISON_COHORT_OBSERVATIONS = 8
TOTAL_UNBALANCED_OBSERVATIONS = 15
MIN_CONVENTIONAL_BIAS = 5.0


def test_clean_room_reference_matches_package_cohort_estimates(
    heterogeneous_effect_panel: pd.DataFrame,
) -> None:
    """Match cohort estimates from an independent subevent-by-subevent oracle."""
    l_min = -2
    l_max = 1
    control_window = 2
    reference_event_time = -1
    result = estimate_stacked_eventstudy(
        data=heterogeneous_effect_panel,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=l_min,
        l_max=l_max,
        control_window=control_window,
        reference_event_time=reference_event_time,
        calendar_year_col="calendar_year",
    )

    oracle = _estimate_clean_room_cohort_effects(
        data=heterogeneous_effect_panel,
        l_min=l_min,
        l_max=l_max,
        control_window=control_window,
        reference_event_time=reference_event_time,
    )
    comparison = result.cohort_params.merge(
        oracle,
        on=["subevent", "event_time"],
        how="inner",
        validate="one_to_one",
    )

    assert comparison.shape[0] == result.cohort_params.shape[0]
    assert np.allclose(
        comparison["estimate"],
        comparison["oracle_estimate"],
        atol=STRICT_ATOL,
        rtol=0.0,
    )


def test_stacked_data_satisfies_paper_invariants(
    two_cohort_panel: pd.DataFrame,
) -> None:
    """Check rolling-control and event-window invariants on returned stacked data."""
    l_min = -2
    l_max = 1
    control_window = 2
    reference_event_time = -1
    result = estimate_stacked_eventstudy(
        data=two_cohort_panel,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=l_min,
        l_max=l_max,
        control_window=control_window,
        reference_event_time=reference_event_time,
        calendar_year_col="calendar_year",
        return_stacked_data=True,
    )
    assert result.stacked_data is not None
    stacked = result.stacked_data

    controls = stacked.loc[stacked["treated_in_subevent"] == 0]
    assert (controls["age"] < controls["treatment_age"]).all()
    assert (controls["treatment_age"] > controls["subevent"]).all()
    assert (controls["treatment_age"] <= controls["subevent"] + control_window).all()

    required_event_times = set(range(l_min, l_max + 1))
    treated_coverage = (
        stacked.loc[stacked["treated_in_subevent"] == 1]
        .groupby("subevent")["event_time"]
        .agg(lambda values: {int(value) for value in values})
    )
    assert all(
        required_event_times.issubset(event_times) for event_times in treated_coverage
    )
    assert reference_event_time not in set(result.cohort_params["event_time"])

    expected_subevents = _get_clean_room_admissible_cohorts(
        data=two_cohort_panel,
        l_min=l_min,
        l_max=l_max,
        control_window=control_window,
    )
    assert set(stacked["subevent"]) == set(expected_subevents)


def test_cohort_weights_use_focal_observation_mass(
    two_cohort_panel: pd.DataFrame,
) -> None:
    """Weight cohorts by focal-cohort rows, not unique focal individuals."""
    first_cohort_unit = int(
        two_cohort_panel.loc[
            two_cohort_panel["treatment_age"] == UNBALANCED_COHORT,
            "id",
        ].min()
    )
    drop_one_focal_row = (
        (two_cohort_panel["treatment_age"] == UNBALANCED_COHORT)
        & (two_cohort_panel["id"] == first_cohort_unit)
        & (two_cohort_panel["age"] == DROPPED_FOCAL_AGE)
    )
    unbalanced = two_cohort_panel.loc[~drop_one_focal_row].copy()

    result = estimate_stacked_eventstudy(
        data=unbalanced,
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
    weights = result.cohort_weights.set_index("subevent")

    assert (
        weights.loc[UNBALANCED_COHORT, "n_observations"]
        == UNBALANCED_COHORT_OBSERVATIONS
    )
    assert (
        weights.loc[COMPARISON_COHORT, "n_observations"]
        == COMPARISON_COHORT_OBSERVATIONS
    )
    assert isclose(
        weights.loc[UNBALANCED_COHORT, "weight_mass"],
        float(UNBALANCED_COHORT_OBSERVATIONS),
        abs_tol=STRICT_ATOL,
    )
    assert isclose(
        weights.loc[COMPARISON_COHORT, "weight_mass"],
        float(COMPARISON_COHORT_OBSERVATIONS),
        abs_tol=STRICT_ATOL,
    )
    assert isclose(
        weights.loc[UNBALANCED_COHORT, "weight"],
        UNBALANCED_COHORT_OBSERVATIONS / TOTAL_UNBALANCED_OBSERVATIONS,
        abs_tol=STRICT_ATOL,
    )
    assert isclose(
        weights.loc[COMPARISON_COHORT, "weight"],
        COMPARISON_COHORT_OBSERVATIONS / TOTAL_UNBALANCED_OBSERVATIONS,
        abs_tol=STRICT_ATOL,
    )


def test_stacked_estimator_recovers_effects_when_conventional_event_study_is_biased() -> (
    None
):
    """Recover known dynamic effects in a design where TWFE event study is biased."""
    data = _make_contaminated_event_study_panel()
    expected = pd.Series(
        {-3: 0.0, -2: 0.0, 0: -10.0, 1: -20.0, 2: -25.0},
        name="expected",
    )

    stacked_result = estimate_stacked_eventstudy(
        data=data,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-3,
        l_max=2,
        control_window=3,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
    )
    stacked_estimates = stacked_result.average_params.set_index("event_time")[
        "estimate"
    ].sort_index()
    conventional_estimates = _estimate_conventional_event_study(
        data=data,
        event_times=tuple(expected.index),
        reference_event_time=-1,
    )

    assert np.allclose(
        stacked_estimates.loc[expected.index],
        expected,
        atol=RECOVERY_ATOL,
        rtol=0.0,
    )
    assert (conventional_estimates - expected).abs().max() > MIN_CONVENTIONAL_BIAS


def _estimate_clean_room_cohort_effects(
    data: pd.DataFrame,
    l_min: int,
    l_max: int,
    control_window: int,
    reference_event_time: int,
) -> pd.DataFrame:
    """Estimate paper subevents without using package internals."""
    rows: list[dict[str, float | int]] = []
    for subevent in _get_clean_room_admissible_cohorts(
        data=data,
        l_min=l_min,
        l_max=l_max,
        control_window=control_window,
    ):
        stack = _build_clean_room_subevent_stack(
            data=data,
            subevent=subevent,
            l_min=l_min,
            l_max=l_max,
            control_window=control_window,
        )
        fit = _fit_clean_room_subevent_model(
            stack=stack,
            reference_event_time=reference_event_time,
        )
        for event_time in range(l_min, l_max + 1):
            if event_time == reference_event_time:
                continue
            term = _clean_room_event_term(event_time=event_time)
            rows.append(
                {
                    "subevent": subevent,
                    "event_time": event_time,
                    "oracle_estimate": float(fit.params[term]),
                }
            )
    return pd.DataFrame(rows).sort_values(["subevent", "event_time"])


def _fit_clean_room_subevent_model(
    stack: pd.DataFrame,
    reference_event_time: int,
) -> object:
    """Fit one clean-room subevent regression."""
    design = stack.copy()
    event_times = sorted(int(value) for value in design["event_time"].unique())
    terms: list[str] = []
    for event_time in event_times:
        if event_time == reference_event_time:
            continue
        term = _clean_room_event_term(event_time=event_time)
        design[term] = (
            (design["focal_cohort"] == 1) & (design["event_time"] == event_time)
        ).astype(int)
        terms.append(term)

    formula = "outcome ~ 0 + " + " + ".join([*terms, "C(id)", "C(age)"])
    return smf.ols(formula=formula, data=design).fit(
        cov_type="cluster",
        cov_kwds={"groups": design["id"], "use_correction": False},
    )


def _build_clean_room_subevent_stack(
    data: pd.DataFrame,
    subevent: int,
    l_min: int,
    l_max: int,
    control_window: int,
) -> pd.DataFrame:
    """Construct one subevent sample from the paper rules."""
    treated = data.loc[data["treatment_age"] == subevent].copy()
    treated["subevent"] = subevent
    treated["event_time"] = treated["age"] - subevent
    treated["focal_cohort"] = 1

    controls = data.loc[
        data["treatment_age"].between(subevent + 1, subevent + control_window)
    ].copy()
    controls = controls.loc[controls["age"] < controls["treatment_age"]]
    controls["subevent"] = subevent
    controls["event_time"] = controls["age"] - subevent
    controls["focal_cohort"] = 0

    stacked = pd.concat([treated, controls], ignore_index=True)
    return stacked.loc[
        (stacked["event_time"] >= l_min) & (stacked["event_time"] <= l_max)
    ].reset_index(drop=True)


def _get_clean_room_admissible_cohorts(
    data: pd.DataFrame,
    l_min: int,
    l_max: int,
    control_window: int,
) -> list[int]:
    """Return cohorts satisfying the paper window and rolling-control rules."""
    observed_min_age = int(data["age"].min())
    max_treatment_age = int(data["treatment_age"].max())
    requested_min_cohort = observed_min_age - l_min
    requested_max_cohort = max_treatment_age - control_window
    required_event_times = set(range(l_min, l_max + 1))
    required_control_event_times = set(range(l_min, min(-1, l_max) + 1))

    cohorts: list[int] = []
    for subevent in sorted(int(value) for value in data["treatment_age"].unique()):
        stack = _build_clean_room_subevent_stack(
            data=data,
            subevent=subevent,
            l_min=l_min,
            l_max=l_max,
            control_window=control_window,
        )
        treated_event_times = {
            int(value)
            for value in stack.loc[stack["focal_cohort"] == 1, "event_time"].unique()
        }
        control_event_times = {
            int(value)
            for value in stack.loc[stack["focal_cohort"] == 0, "event_time"].unique()
        }
        in_requested_range = requested_min_cohort <= subevent <= requested_max_cohort
        if (
            in_requested_range
            and required_event_times.issubset(treated_event_times)
            and required_control_event_times.issubset(control_event_times)
        ):
            cohorts.append(subevent)
    return cohorts


def _make_contaminated_event_study_panel() -> pd.DataFrame:
    """Create a panel where already-treated comparisons contaminate TWFE."""
    dynamic_effects = {-3: 0.0, -2: 0.0, -1: 0.0, 0: -10.0, 1: -20.0, 2: -25.0}
    rows: list[dict[str, float | int]] = []
    unit_id = 1
    for treatment_age in range(25, 33):
        for _replicate in range(10):
            for age in range(22, 36):
                event_time = age - treatment_age
                treatment_effect = _get_dynamic_effect(
                    event_time=event_time,
                    dynamic_effects=dynamic_effects,
                )
                rows.append(
                    {
                        "id": unit_id,
                        "age": age,
                        "treatment_age": treatment_age,
                        "calendar_year": 2000 + age,
                        "outcome": 100.0 + 2.0 * age + treatment_effect,
                    }
                )
            unit_id += 1
    return pd.DataFrame(rows)


def _get_dynamic_effect(
    event_time: int,
    dynamic_effects: dict[int, float],
) -> float:
    """Return a dynamic effect with a flat long-run tail."""
    if event_time in dynamic_effects:
        return dynamic_effects[event_time]
    if event_time > max(dynamic_effects):
        return dynamic_effects[max(dynamic_effects)]
    return 0.0


def _estimate_conventional_event_study(
    data: pd.DataFrame,
    event_times: tuple[int, ...],
    reference_event_time: int,
) -> pd.Series:
    """Estimate a conventional TWFE event-study specification."""
    design = data.copy()
    terms: list[str] = []
    relative_time = design["age"] - design["treatment_age"]
    for event_time in event_times:
        if event_time == reference_event_time:
            continue
        term = _conventional_event_term(event_time=event_time)
        design[term] = relative_time.eq(event_time).astype(int)
        terms.append(term)

    formula = "outcome ~ 0 + " + " + ".join([*terms, "C(id)", "C(age)"])
    fit = smf.ols(formula=formula, data=design).fit()
    return pd.Series(
        {
            event_time: float(fit.params[_conventional_event_term(event_time)])
            for event_time in event_times
            if event_time != reference_event_time
        },
    ).sort_index()


def _clean_room_event_term(event_time: int) -> str:
    """Return a clean-room event-study term name."""
    return f"oracle_l_{_encode_event_time(event_time=event_time)}"


def _conventional_event_term(event_time: int) -> str:
    """Return a conventional event-study term name."""
    return f"conventional_l_{_encode_event_time(event_time=event_time)}"


def _encode_event_time(event_time: int) -> str:
    """Encode relative time as a valid column-name suffix."""
    prefix = "m" if event_time < 0 else "p"
    return f"{prefix}{abs(event_time)}"
