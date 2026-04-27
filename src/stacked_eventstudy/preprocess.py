"""Data preprocessing for estimation."""

from collections.abc import Sequence

import pandas as pd

from stacked_eventstudy.types import EstimatorConfig


def prepare_panel_data(data: pd.DataFrame, config: EstimatorConfig) -> pd.DataFrame:
    """Normalize input data to a canonical internal representation."""
    selected_columns = [
        config.id_col,
        config.age_col,
        config.treatment_age_col,
        config.outcome_col,
        *config.covariates,
    ]
    if config.calendar_year_col is not None:
        selected_columns.append(config.calendar_year_col)
    if config.weights_col is not None:
        selected_columns.append(config.weights_col)
    if config.cluster_col is not None and config.cluster_col not in selected_columns:
        selected_columns.append(config.cluster_col)

    panel = data.loc[:, selected_columns].copy()
    renamed_columns = {
        config.id_col: "unit_id",
        config.age_col: "age",
        config.treatment_age_col: "treatment_age",
        config.outcome_col: "outcome",
    }
    if config.calendar_year_col is not None:
        renamed_columns[config.calendar_year_col] = "calendar_year"
    if config.weights_col is not None:
        renamed_columns[config.weights_col] = "input_weight"
    if config.cluster_col is not None:
        renamed_columns[config.cluster_col] = "cluster_id"

    panel = panel.rename(columns=renamed_columns)

    if config.cluster_col is None:
        panel["cluster_id"] = panel["unit_id"]

    if config.weights_col is None:
        panel["input_weight"] = 1.0

    panel["event_time_own"] = panel["age"] - panel["treatment_age"]
    panel = panel.sort_values(["unit_id", "age"]).reset_index(drop=True)
    return panel


def keep_admissible_cohorts(
    data: pd.DataFrame,
    admissible_cohorts: Sequence[int],
) -> pd.DataFrame:
    """Restrict the sample to admissible treated cohorts."""
    return data.loc[data["treatment_age"].isin(admissible_cohorts)].reset_index(drop=True)
