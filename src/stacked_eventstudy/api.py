"""Public API for the stacked estimator."""

from collections.abc import Sequence

import pandas as pd

from stacked_eventstudy.aggregation import aggregate_cohort_params, compute_cohort_weights
from stacked_eventstudy.estimation import (
    estimate_joint_stacked_model,
    extract_cohort_params_from_joint_model,
    extract_joint_parameter_covariance,
)
from stacked_eventstudy.preprocess import prepare_panel_data
from stacked_eventstudy.scaling import (
    compute_pre_birth_levels,
    scale_cohort_params,
    scale_covariance_by_event_time,
)
from stacked_eventstudy.stacking import build_stacked_data
from stacked_eventstudy.types import EstimatorConfig, StackedEventStudyResult
from stacked_eventstudy.utils import coerce_covariates
from stacked_eventstudy.validate import _validate_with_config, validate_stacked_eventstudy


def estimate_stacked_eventstudy(
    data: pd.DataFrame,
    id_col: str,
    age_col: str,
    treatment_age_col: str,
    outcome_col: str,
    l_min: int = -3,
    l_max: int = 4,
    control_window: int = 5,
    reference_event_time: int = -1,
    min_treatment_age: int | None = None,
    max_treatment_age: int | None = None,
    observed_min_age: int | None = None,
    calendar_year_col: str | None = None,
    covariates: Sequence[str] = (),
    weights_col: str | None = None,
    cluster_col: str | None = None,
    balance: bool = True,
    scale: str = "none",
    backend: str = "statsmodels",
    return_stacked_data: bool = False,
) -> StackedEventStudyResult:
    """Estimate stacked event-study effects with rolling-window controls.

    Args:
        data: Individual-level panel data.
        id_col: Column containing the individual identifier.
        age_col: Column containing age in integer years.
        treatment_age_col: Column containing age at first birth.
        outcome_col: Column containing the outcome variable.
        l_min: Minimum event time to include.
        l_max: Maximum event time to include.
        control_window: Number of future-treated cohorts used as controls.
        reference_event_time: Omitted event time in the event-study design.
        min_treatment_age: Optional lower bound on treated cohorts to estimate.
        max_treatment_age: Optional upper bound on treated cohorts to estimate.
        observed_min_age: Optional minimum observed age used in feasibility checks.
        calendar_year_col: Optional calendar-year column.
        covariates: Optional additional covariate columns.
        weights_col: Optional observation-weight column.
        cluster_col: Optional clustering column. Defaults to the original individual id.
        balance: Whether to require complete treated and control support in the
            requested window.
        scale: Whether to return raw effects or pre-birth scaled effects.
        backend: Regression backend name. The current implementation uses
            `statsmodels`.
        return_stacked_data: Whether to return the constructed stacked sample.

    Returns:
        A `StackedEventStudyResult` containing cohort-specific effects, aggregated
        effects, cohort weights, validation output, and optionally the stacked data.

    Raises:
        ValueError: If the input data fails validation.
    """
    config = EstimatorConfig(
        id_col=id_col,
        age_col=age_col,
        treatment_age_col=treatment_age_col,
        outcome_col=outcome_col,
        l_min=l_min,
        l_max=l_max,
        control_window=control_window,
        reference_event_time=reference_event_time,
        min_treatment_age=min_treatment_age,
        max_treatment_age=max_treatment_age,
        observed_min_age=observed_min_age,
        calendar_year_col=calendar_year_col,
        covariates=coerce_covariates(covariates),
        weights_col=weights_col,
        cluster_col=cluster_col,
        balance=balance,
        scale=scale,
        backend=backend,
        return_stacked_data=return_stacked_data,
    )
    validation = _validate_with_config(data=data, config=config)
    if not validation.is_valid:
        msg = "Input validation failed: " + " ".join(validation.errors)
        raise ValueError(msg)

    panel = prepare_panel_data(data=data, config=config)
    stacked_data = build_stacked_data(data=panel, config=config, validation=validation)

    joint_model = estimate_joint_stacked_model(stacked_data=stacked_data, config=config)
    cohort_params = extract_cohort_params_from_joint_model(
        fitted_model=joint_model,
        stacked_data=stacked_data,
        config=config,
    )
    parameter_covariance = extract_joint_parameter_covariance(
        fitted_model=joint_model,
        cohort_params=cohort_params,
        config=config,
    )
    cohort_weights = compute_cohort_weights(stacked_data=stacked_data)
    cohort_counts = validation.cohort_diagnostics.loc[
        validation.cohort_diagnostics["admissible"],
        [
            "subevent",
            "n_treated_individuals",
            "n_control_individuals",
            "n_treated_obs",
            "n_control_obs",
        ],
    ]

    if scale == "pre_birth":
        pre_birth_levels = compute_pre_birth_levels(data=panel, config=config)
        cohort_params = scale_cohort_params(
            cohort_params=cohort_params,
            pre_birth_levels=pre_birth_levels,
        )
        parameter_covariance = scale_covariance_by_event_time(
            parameter_covariance=parameter_covariance,
            pre_birth_levels=pre_birth_levels,
        )

    cohort_params = cohort_params.merge(
        cohort_counts,
        on="subevent",
        how="left",
        validate="many_to_one",
    )

    average_params, vcov_average = aggregate_cohort_params(
        cohort_params=cohort_params,
        cohort_weights=cohort_weights,
        parameter_covariance=parameter_covariance,
    )

    if scale == "pre_birth":
        average_params["scale"] = "pre_birth"

    return StackedEventStudyResult(
        cohort_params=cohort_params,
        average_params=average_params,
        cohort_weights=cohort_weights,
        vcov_average=vcov_average,
        config=config,
        model_summaries={"joint": joint_model},
        validation=validation,
        stacked_data=stacked_data if return_stacked_data else None,
    )


__all__ = ["estimate_stacked_eventstudy", "validate_stacked_eventstudy"]
