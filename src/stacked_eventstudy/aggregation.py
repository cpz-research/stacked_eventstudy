"""Aggregation helpers."""

import pandas as pd

from stacked_eventstudy.utils import (
    make_confidence_interval,
    standard_error_from_variance,
)


def compute_cohort_weights(stacked_data: pd.DataFrame) -> pd.DataFrame:
    """Compute treated-cohort weights using treated individual counts."""
    treated_counts = (
        stacked_data.loc[
            stacked_data["treated_in_subevent"] == 1, ["subevent", "unit_id"]
        ]
        .drop_duplicates()
        .groupby("subevent", as_index=False)
        .size()
        .rename(columns={"size": "n_individuals"})
    )
    total_individuals = int(treated_counts["n_individuals"].sum())
    treated_counts["weight"] = treated_counts["n_individuals"] / total_individuals
    return treated_counts.sort_values("subevent").reset_index(drop=True)


def aggregate_cohort_params(
    cohort_params: pd.DataFrame,
    cohort_weights: pd.DataFrame,
    covariance_by_event_time: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate cohort-specific effects using cohort weights."""
    average_rows: list[dict[str, object]] = []
    event_times = sorted(int(value) for value in cohort_params["event_time"].unique())
    vcov_rows: dict[int, dict[int, float]] = {
        event_time: {} for event_time in event_times
    }

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
        covariance_block = covariance_by_event_time[event_time]
        aligned_weights = event_params.set_index("subevent")["weight"].reindex(
            covariance_block.index,
        )
        variance = float(
            aligned_weights.to_numpy()
            @ covariance_block.to_numpy()
            @ aligned_weights.to_numpy()
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
        for other_event_time in event_times:
            vcov_rows[event_time][other_event_time] = 0.0
        vcov_rows[event_time][event_time] = variance

    average_params = (
        pd.DataFrame(average_rows).sort_values("event_time").reset_index(drop=True)
    )
    vcov_average = pd.DataFrame(vcov_rows).sort_index().sort_index(axis=1)
    return average_params, vcov_average
