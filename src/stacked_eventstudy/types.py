"""Shared typed objects."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EstimatorConfig:
    """Store normalized estimator settings used across the package."""

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
    heterogeneity_col: str | None
    heterogeneity_weighting: str
    balance: bool
    scale: str
    backend: str
    return_stacked_data: bool


@dataclass(frozen=True)
class StackedEventStudyValidation:
    """Store validation diagnostics for stacked event-study estimation.

    Attributes:
        is_valid: Whether the requested design passed validation.
        errors: Fatal validation problems.
        warnings: Non-fatal diagnostics worth surfacing to the user.
        observed_min_age: Minimum age used in feasibility calculations.
        requested_cohort_range: Cohort range implied by the requested settings.
        admissible_cohort_range: Cohort range that remains after restrictions.
        cohort_diagnostics: Cohort-by-cohort feasibility table.
        sample_counts: Sample-level counts used in diagnostics.
        window_feasibility: Requested window checks and their pass/fail status.
    """

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
    """Store the outputs of stacked event-study estimation.

    Attributes:
        cohort_params: Cohort-specific event-study coefficients.
        average_params: Aggregated event-study coefficients by event time.
        cohort_weights: Treated cohort weights used in aggregation.
        vcov_average: Covariance matrix for aggregated effects.
        contrast_params: Pairwise contrasts across heterogeneity groups. Empty when
            no heterogeneity column is supplied.
        config: Normalized estimator settings.
        model_summaries: Backend model results returned by the estimator.
        validation: Validation output used during estimation.
        stacked_data: Constructed stacked sample when requested.
    """

    cohort_params: pd.DataFrame
    average_params: pd.DataFrame
    cohort_weights: pd.DataFrame
    vcov_average: pd.DataFrame
    contrast_params: pd.DataFrame
    config: EstimatorConfig
    model_summaries: dict[str, object]
    validation: StackedEventStudyValidation
    stacked_data: pd.DataFrame | None
