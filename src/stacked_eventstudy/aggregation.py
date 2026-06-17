"""Aggregation helpers."""

from itertools import combinations
from math import erfc, sqrt

import pandas as pd

from stacked_eventstudy.utils import (
    make_confidence_interval,
    standard_error_from_variance,
)

CONTRAST_COLUMNS = [
    "heterogeneity_col",
    "group_left",
    "group_right",
    "event_time",
    "weight_scheme",
    "estimate",
    "std_error",
    "ci_low",
    "ci_high",
    "p_value",
]


def compute_cohort_weights(stacked_data: pd.DataFrame) -> pd.DataFrame:
    """Compute treated-cohort weights using focal treated observation mass."""
    treated_observations = (
        stacked_data.loc[
            stacked_data["treated_in_subevent"] == 1, ["subevent", "input_weight"]
        ]
        .groupby("subevent", as_index=False)
        .agg(
            n_observations=("input_weight", "size"),
            weight_mass=("input_weight", "sum"),
        )
    )
    total_mass = float(treated_observations["weight_mass"].sum())
    treated_observations["weight"] = treated_observations["weight_mass"] / total_mass
    return treated_observations.sort_values("subevent").reset_index(drop=True)


def compute_heterogeneous_cohort_weights(
    stacked_data: pd.DataFrame,
    weight_scheme: str,
) -> pd.DataFrame:
    """Compute cohort weights by heterogeneity group."""
    if weight_scheme == "within":
        return _compute_within_heterogeneity_cohort_weights(stacked_data=stacked_data)
    if weight_scheme == "overall":
        return _compute_overall_heterogeneity_cohort_weights(stacked_data=stacked_data)
    msg = f"Unsupported heterogeneity weighting: {weight_scheme}."
    raise ValueError(msg)


def compute_cohort_counts(stacked_data: pd.DataFrame) -> pd.DataFrame:
    """Compute treated and control counts by cohort from the actual stack."""
    return _compute_counts_by_keys(stacked_data=stacked_data, keys=["subevent"])


def compute_heterogeneous_cohort_counts(stacked_data: pd.DataFrame) -> pd.DataFrame:
    """Compute cohort counts by heterogeneity group."""
    return _compute_counts_by_keys(
        stacked_data=stacked_data,
        keys=["heterogeneity_value", "subevent"],
    )


def _compute_counts_by_keys(
    stacked_data: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Compute treated and control counts by grouping keys."""
    count_columns = [
        "n_treated_individuals",
        "n_control_individuals",
        "n_treated_obs",
        "n_control_obs",
    ]
    treated = stacked_data.loc[stacked_data["treated_in_subevent"] == 1]
    controls = stacked_data.loc[stacked_data["treated_in_subevent"] == 0]

    treated_individuals = _count_unique_units(
        data=treated,
        keys=keys,
        column_name="n_treated_individuals",
    )
    control_individuals = _count_unique_units(
        data=controls,
        keys=keys,
        column_name="n_control_individuals",
    )
    treated_observations = _count_rows(
        data=treated,
        keys=keys,
        column_name="n_treated_obs",
    )
    control_observations = _count_rows(
        data=controls,
        keys=keys,
        column_name="n_control_obs",
    )
    counts = (
        treated_individuals.merge(
            control_individuals,
            on=keys,
            how="outer",
            validate="one_to_one",
        )
        .merge(treated_observations, on=keys, how="outer", validate="one_to_one")
        .merge(control_observations, on=keys, how="outer", validate="one_to_one")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    counts[count_columns] = counts[count_columns].fillna(0).astype(int)
    return counts


def aggregate_cohort_params(
    cohort_params: pd.DataFrame,
    cohort_weights: pd.DataFrame,
    parameter_covariance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate cohort-specific effects using cohort weights."""
    average_rows: list[dict[str, object]] = []
    event_times = sorted(int(value) for value in cohort_params["event_time"].unique())
    vcov_rows: dict[int, dict[int, float]] = {
        event_time: {} for event_time in event_times
    }

    weight_vectors: dict[int, pd.Series] = {}
    covariance_blocks: dict[tuple[int, int], pd.DataFrame] = {}

    for event_time in event_times:
        event_params = cohort_params.loc[
            cohort_params["event_time"] == event_time
        ].merge(
            cohort_weights,
            on="subevent",
            how="left",
            validate="many_to_one",
        )
        estimate = float((event_params["estimate"] * event_params["weight"]).sum())
        covariance_block = parameter_covariance.loc[
            (event_time, slice(None)),
            (event_time, slice(None)),
        ]
        covariance_block.index = covariance_block.index.get_level_values("subevent")
        covariance_block.columns = covariance_block.columns.get_level_values("subevent")
        aligned_weights = event_params.set_index("subevent")["weight"].reindex(
            covariance_block.index
        )
        variance = float(
            aligned_weights.to_numpy()
            @ covariance_block.to_numpy()
            @ aligned_weights.to_numpy(),
        )
        std_error = standard_error_from_variance(variance)
        ci_low, ci_high = make_confidence_interval(
            estimate=estimate, std_error=std_error
        )
        average_rows.append(
            {
                "event_time": event_time,
                "estimate": estimate,
                "std_error": std_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_cohorts": int(event_params["subevent"].nunique()),
                "scale": str(event_params["scale"].iloc[0]),
            },
        )
        weight_vectors[event_time] = aligned_weights
        covariance_blocks[(event_time, event_time)] = covariance_block

    average_params = (
        pd.DataFrame(average_rows).sort_values("event_time").reset_index(drop=True)
    )
    for event_time in event_times:
        for other_event_time in event_times:
            if (event_time, other_event_time) not in covariance_blocks:
                covariance_block = parameter_covariance.loc[
                    (event_time, slice(None)),
                    (other_event_time, slice(None)),
                ]
                covariance_block.index = covariance_block.index.get_level_values(
                    "subevent"
                )
                covariance_block.columns = covariance_block.columns.get_level_values(
                    "subevent"
                )
                covariance_blocks[(event_time, other_event_time)] = covariance_block
            left_weights = weight_vectors[event_time].reindex(
                covariance_blocks[(event_time, other_event_time)].index
            )
            right_weights = weight_vectors[other_event_time].reindex(
                covariance_blocks[(event_time, other_event_time)].columns
            )
            covariance_value = float(
                left_weights.to_numpy()
                @ covariance_blocks[(event_time, other_event_time)].to_numpy()
                @ right_weights.to_numpy(),
            )
            vcov_rows[event_time][other_event_time] = covariance_value
    vcov_average = pd.DataFrame(vcov_rows).sort_index().sort_index(axis=1)
    return average_params, vcov_average


def aggregate_heterogeneous_cohort_params(
    cohort_params: pd.DataFrame,
    cohort_weights: pd.DataFrame,
    parameter_covariance: pd.DataFrame,
    heterogeneity_col: str,
    weight_scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate cohort-specific effects by heterogeneity group."""
    average_rows: list[dict[str, object]] = []
    heterogeneity_values = _sorted_unique_values(cohort_params["heterogeneity_value"])
    event_times = sorted(int(value) for value in cohort_params["event_time"].unique())
    weight_vectors: dict[tuple[object, int], pd.Series] = {}
    covariance_blocks: dict[tuple[object, int, object, int], pd.DataFrame] = {}

    for heterogeneity_value in heterogeneity_values:
        for event_time in event_times:
            event_params = _get_heterogeneous_event_params(
                cohort_params=cohort_params,
                cohort_weights=cohort_weights,
                heterogeneity_value=heterogeneity_value,
                event_time=event_time,
            )
            covariance_block = _get_heterogeneous_covariance_block(
                parameter_covariance=parameter_covariance,
                left_value=heterogeneity_value,
                left_event_time=event_time,
                right_value=heterogeneity_value,
                right_event_time=event_time,
            )
            aligned_weights = event_params.set_index("subevent")["weight"].reindex(
                covariance_block.index
            )
            estimate = float((event_params["estimate"] * event_params["weight"]).sum())
            variance = float(
                aligned_weights.to_numpy()
                @ covariance_block.to_numpy()
                @ aligned_weights.to_numpy(),
            )
            std_error = standard_error_from_variance(variance)
            ci_low, ci_high = make_confidence_interval(
                estimate=estimate,
                std_error=std_error,
            )
            average_rows.append(
                {
                    "heterogeneity_col": heterogeneity_col,
                    "heterogeneity_value": heterogeneity_value,
                    "event_time": event_time,
                    "estimate": estimate,
                    "std_error": std_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_cohorts": int(event_params["subevent"].nunique()),
                    "scale": str(event_params["scale"].iloc[0]),
                    "weight_scheme": weight_scheme,
                },
            )
            weight_vectors[(heterogeneity_value, event_time)] = aligned_weights
            covariance_blocks[
                (
                    heterogeneity_value,
                    event_time,
                    heterogeneity_value,
                    event_time,
                )
            ] = covariance_block

    average_params = (
        pd.DataFrame(average_rows)
        .sort_values(["heterogeneity_value", "event_time"])
        .reset_index(drop=True)
    )
    vcov_average = _build_heterogeneous_average_covariance(
        heterogeneity_values=heterogeneity_values,
        event_times=event_times,
        parameter_covariance=parameter_covariance,
        weight_vectors=weight_vectors,
        covariance_blocks=covariance_blocks,
    )
    contrast_params = _build_contrast_params(
        average_params=average_params,
        vcov_average=vcov_average,
        heterogeneity_col=heterogeneity_col,
        weight_scheme=weight_scheme,
    )
    return average_params, vcov_average, contrast_params


def make_empty_contrast_params() -> pd.DataFrame:
    """Return an empty pairwise contrast table."""
    return pd.DataFrame(columns=CONTRAST_COLUMNS)


def _compute_within_heterogeneity_cohort_weights(
    stacked_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute focal observation-mass weights within each heterogeneity group."""
    treated_observations = (
        stacked_data.loc[
            stacked_data["treated_in_subevent"] == 1,
            ["heterogeneity_value", "subevent", "input_weight"],
        ]
        .groupby(["heterogeneity_value", "subevent"], as_index=False)
        .agg(
            n_observations=("input_weight", "size"),
            weight_mass=("input_weight", "sum"),
        )
    )
    totals = treated_observations.groupby("heterogeneity_value")[
        "weight_mass"
    ].transform("sum")
    treated_observations["weight"] = treated_observations["weight_mass"] / totals
    treated_observations["weight_scheme"] = "within"
    return treated_observations.sort_values(
        ["heterogeneity_value", "subevent"]
    ).reset_index(
        drop=True,
    )


def _compute_overall_heterogeneity_cohort_weights(
    stacked_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compute common cohort weights for each heterogeneity group."""
    overall_weights = compute_cohort_weights(stacked_data=stacked_data)
    heterogeneity_values = pd.DataFrame(
        {
            "heterogeneity_value": _sorted_unique_values(
                stacked_data["heterogeneity_value"],
            ),
        },
    )
    weights = heterogeneity_values.merge(overall_weights, how="cross")
    weights["weight_scheme"] = "overall"
    return weights.sort_values(["heterogeneity_value", "subevent"]).reset_index(
        drop=True
    )


def _count_unique_units(
    data: pd.DataFrame,
    keys: list[str],
    column_name: str,
) -> pd.DataFrame:
    """Count unique units by keys."""
    return (
        data.loc[:, [*keys, "unit_id"]]
        .drop_duplicates()
        .groupby(keys, as_index=False)
        .size()
        .rename(columns={"size": column_name})
    )


def _count_rows(data: pd.DataFrame, keys: list[str], column_name: str) -> pd.DataFrame:
    """Count rows by keys."""
    return (
        data.groupby(keys, as_index=False).size().rename(columns={"size": column_name})
    )


def _get_heterogeneous_event_params(
    cohort_params: pd.DataFrame,
    cohort_weights: pd.DataFrame,
    heterogeneity_value: object,
    event_time: int,
) -> pd.DataFrame:
    """Return event-time rows merged with heterogeneity-specific weights."""
    event_params = cohort_params.loc[
        (cohort_params["heterogeneity_value"] == heterogeneity_value)
        & (cohort_params["event_time"] == event_time)
    ].merge(
        cohort_weights,
        on=["heterogeneity_value", "subevent"],
        how="left",
        validate="many_to_one",
    )
    if event_params["weight"].isna().any():
        msg = "Missing heterogeneity cohort weights."
        raise ValueError(msg)
    event_params = event_params.rename(columns={"weight": "raw_weight"})
    raw_weight_sum = float(event_params["raw_weight"].sum())
    if raw_weight_sum <= 0:
        msg = "Heterogeneity cohort weights must have positive available weight."
        raise ValueError(msg)
    event_params["weight"] = event_params["raw_weight"] / raw_weight_sum
    return event_params


def _get_heterogeneous_covariance_block(
    parameter_covariance: pd.DataFrame,
    left_value: object,
    left_event_time: int,
    right_value: object,
    right_event_time: int,
) -> pd.DataFrame:
    """Extract a group-event covariance block indexed by cohort."""
    covariance_block = parameter_covariance.loc[
        (left_value, left_event_time, slice(None)),
        (right_value, right_event_time, slice(None)),
    ]
    covariance_block.index = covariance_block.index.get_level_values("subevent")
    covariance_block.columns = covariance_block.columns.get_level_values("subevent")
    return covariance_block


def _build_heterogeneous_average_covariance(
    heterogeneity_values: list[object],
    event_times: list[int],
    parameter_covariance: pd.DataFrame,
    weight_vectors: dict[tuple[object, int], pd.Series],
    covariance_blocks: dict[tuple[object, int, object, int], pd.DataFrame],
) -> pd.DataFrame:
    """Build covariance matrix for heterogeneity-specific average effects."""
    index_keys = [
        (heterogeneity_value, event_time)
        for heterogeneity_value in heterogeneity_values
        for event_time in event_times
    ]
    vcov_rows: dict[tuple[object, int], dict[tuple[object, int], float]] = {
        key: {} for key in index_keys
    }
    for left_value, left_event_time in index_keys:
        for right_value, right_event_time in index_keys:
            block_key = (left_value, left_event_time, right_value, right_event_time)
            if block_key not in covariance_blocks:
                covariance_blocks[block_key] = _get_heterogeneous_covariance_block(
                    parameter_covariance=parameter_covariance,
                    left_value=left_value,
                    left_event_time=left_event_time,
                    right_value=right_value,
                    right_event_time=right_event_time,
                )
            covariance_block = covariance_blocks[block_key]
            left_weights = weight_vectors[(left_value, left_event_time)].reindex(
                covariance_block.index
            )
            right_weights = weight_vectors[(right_value, right_event_time)].reindex(
                covariance_block.columns
            )
            covariance_value = float(
                left_weights.to_numpy()
                @ covariance_block.to_numpy()
                @ right_weights.to_numpy(),
            )
            vcov_rows[(left_value, left_event_time)][
                (right_value, right_event_time)
            ] = covariance_value
    vcov_average = pd.DataFrame(vcov_rows).sort_index().sort_index(axis=1)
    vcov_average.index = pd.MultiIndex.from_tuples(
        vcov_average.index,
        names=["heterogeneity_value", "event_time"],
    )
    vcov_average.columns = pd.MultiIndex.from_tuples(
        vcov_average.columns,
        names=["heterogeneity_value", "event_time"],
    )
    return vcov_average


def _build_contrast_params(
    average_params: pd.DataFrame,
    vcov_average: pd.DataFrame,
    heterogeneity_col: str,
    weight_scheme: str,
) -> pd.DataFrame:
    """Build pairwise contrasts across heterogeneity groups."""
    rows: list[dict[str, object]] = []
    heterogeneity_values = _sorted_unique_values(average_params["heterogeneity_value"])
    event_times = sorted(int(value) for value in average_params["event_time"].unique())
    estimates = average_params.set_index(["heterogeneity_value", "event_time"])[
        "estimate"
    ]
    for left_value, right_value in combinations(heterogeneity_values, 2):
        for event_time in event_times:
            left_key = (left_value, event_time)
            right_key = (right_value, event_time)
            estimate = float(estimates.loc[left_key] - estimates.loc[right_key])
            variance = float(
                vcov_average.loc[left_key, left_key]
                + vcov_average.loc[right_key, right_key]
                - 2 * vcov_average.loc[left_key, right_key]
            )
            std_error = standard_error_from_variance(variance)
            ci_low, ci_high = make_confidence_interval(
                estimate=estimate,
                std_error=std_error,
            )
            rows.append(
                {
                    "heterogeneity_col": heterogeneity_col,
                    "group_left": left_value,
                    "group_right": right_value,
                    "event_time": event_time,
                    "weight_scheme": weight_scheme,
                    "estimate": estimate,
                    "std_error": std_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": _normal_two_sided_p_value(
                        estimate=estimate,
                        std_error=std_error,
                    ),
                },
            )
    if not rows:
        return make_empty_contrast_params()
    return (
        pd.DataFrame(rows)
        .sort_values(["group_left", "group_right", "event_time"])
        .reset_index(drop=True)
    )


def _normal_two_sided_p_value(estimate: float, std_error: float) -> float:
    """Return a normal-approximation two-sided p-value."""
    if std_error == 0:
        return 1.0 if estimate == 0 else 0.0
    return erfc(abs(estimate / std_error) / sqrt(2.0))


def _sorted_unique_values(series: pd.Series) -> list[object]:
    """Return deterministic sorted unique values."""
    values = series.drop_duplicates().tolist()
    return sorted(values, key=lambda value: (str(type(value)), str(value)))
