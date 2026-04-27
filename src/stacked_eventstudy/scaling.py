"""Scaling helpers."""

import pandas as pd

from stacked_eventstudy.types import EstimatorConfig


def compute_pre_birth_levels(data: pd.DataFrame, config: EstimatorConfig) -> pd.DataFrame:
    """Compute cohort-specific pre-birth outcome levels for scaling."""
    treated_reference = data.loc[data["event_time_own"] == config.reference_event_time]
    levels = (
        treated_reference.groupby("treatment_age", as_index=False)["outcome"]
        .mean()
        .rename(columns={"treatment_age": "subevent", "outcome": "pre_birth_level"})
    )
    if (levels["pre_birth_level"].abs() < 1e-12).any():
        msg = "At least one cohort has a zero or near-zero pre-birth level."
        raise ValueError(msg)
    return levels


def scale_cohort_params(
    cohort_params: pd.DataFrame,
    pre_birth_levels: pd.DataFrame,
) -> pd.DataFrame:
    """Scale cohort-level effects by pre-birth levels."""
    merged = cohort_params.merge(pre_birth_levels, on="subevent", how="left", validate="many_to_one")
    scaled = merged.copy()
    for column in ["estimate", "std_error", "ci_low", "ci_high"]:
        scaled[column] = scaled[column] / scaled["pre_birth_level"]
    scaled["scale"] = "pre_birth"
    return scaled.drop(columns="pre_birth_level")


def scale_covariance_by_event_time(
    parameter_covariance: pd.DataFrame,
    pre_birth_levels: pd.DataFrame,
) -> pd.DataFrame:
    """Scale the joint covariance matrix using cohort-specific pre-birth levels."""
    denominator_map = pre_birth_levels.set_index("subevent")["pre_birth_level"]
    row_subevents = parameter_covariance.index.get_level_values("subevent")
    column_subevents = parameter_covariance.columns.get_level_values("subevent")
    row_denominators = denominator_map.reindex(row_subevents).to_numpy()
    column_denominators = denominator_map.reindex(column_subevents).to_numpy()
    scale_matrix = pd.DataFrame(
        1.0 / (row_denominators[:, None] * column_denominators[None, :]),
        index=parameter_covariance.index,
        columns=parameter_covariance.columns,
    )
    return parameter_covariance * scale_matrix
