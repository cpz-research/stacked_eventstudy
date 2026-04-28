"""Data preprocessing for estimation."""

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
    for column in (
        config.calendar_year_col,
        config.heterogeneity_col,
        config.weights_col,
        config.cluster_col,
    ):
        _append_column_if_present(columns=selected_columns, column=column)

    panel = data.loc[:, selected_columns].copy()
    renamed_columns = {
        config.id_col: "unit_id",
        config.age_col: "age",
        config.treatment_age_col: "treatment_age",
        config.outcome_col: "outcome",
    }
    _add_optional_rename(
        renamed_columns=renamed_columns,
        source=config.calendar_year_col,
        target="calendar_year",
    )
    _add_optional_rename(
        renamed_columns=renamed_columns,
        source=config.heterogeneity_col,
        target="heterogeneity_value",
    )
    _add_optional_rename(
        renamed_columns=renamed_columns,
        source=config.weights_col,
        target="input_weight",
    )
    _add_optional_rename(
        renamed_columns=renamed_columns,
        source=config.cluster_col,
        target="cluster_id",
    )

    panel = panel.rename(columns=renamed_columns)

    if config.cluster_col is None:
        panel["cluster_id"] = panel["unit_id"]

    if config.weights_col is None:
        panel["input_weight"] = 1.0

    panel["event_time_own"] = panel["age"] - panel["treatment_age"]
    return panel.sort_values(["unit_id", "age"]).reset_index(drop=True)


def _append_column_if_present(columns: list[str], column: str | None) -> None:
    """Append a non-missing column once."""
    if column is not None and column not in columns:
        columns.append(column)


def _add_optional_rename(
    renamed_columns: dict[str, str],
    source: str | None,
    target: str,
) -> None:
    """Add a rename entry when the source column exists."""
    if source is not None:
        renamed_columns[source] = target
