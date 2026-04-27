"""Shared typed objects."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EstimatorConfig:
    """Normalized estimator configuration."""

    id_col: str
    age_col: str
    treatment_age_col: str
    outcome_col: str
    l_min: int
    l_max: int
    control_window: int
    reference_event_time: int
    min_treatment_age: int | None
    max_treatment_age: int | None
    observed_min_age: int | None
    calendar_year_col: str | None
    covariates: tuple[str, ...]
    weights_col: str | None
    cluster_col: str | None
    balance: bool
    scale: str
    backend: str
    return_stacked_data: bool


@dataclass(frozen=True)
class StackedEventStudyValidation:
    """Validation diagnostics for stacked event-study estimation."""

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    observed_min_age: int
    requested_cohort_range: tuple[int, int] | None
    admissible_cohort_range: tuple[int, int] | None
    cohort_diagnostics: pd.DataFrame
    sample_counts: pd.DataFrame
    window_feasibility: pd.DataFrame


@dataclass(frozen=True)
class StackedEventStudyResult:
    """Result object for the stacked event-study estimator."""

    cohort_params: pd.DataFrame
    average_params: pd.DataFrame
    cohort_weights: pd.DataFrame
    vcov_average: pd.DataFrame
    config: EstimatorConfig
    model_summaries: dict[int, object]
    validation: StackedEventStudyValidation
    stacked_data: pd.DataFrame | None
