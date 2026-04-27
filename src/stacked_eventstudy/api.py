"""Public API for the stacked estimator."""

from collections.abc import Sequence

import pandas as pd

from stacked_eventstudy.aggregation import (
    aggregate_cohort_params,
    compute_cohort_weights,
)
from stacked_eventstudy.estimation import (
    estimate_cohort_models,
    estimate_joint_stacked_model,
    extract_joint_covariance_by_event_time,
)
from stacked_eventstudy.preprocess import keep_admissible_cohorts, prepare_panel_data
from stacked_eventstudy.scaling import (
    compute_pre_birth_levels,
    scale_cohort_params,
    scale_covariance_by_event_time,
)
from stacked_eventstudy.stacking import build_stacked_data
from stacked_eventstudy.types import EstimatorConfig, StackedEventStudyResult
from stacked_eventstudy.utils import coerce_covariates
from stacked_eventstudy.validate import (
    _validate_with_config,
    validate_stacked_eventstudy,
)


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
    """Estimate the stacked event-study design."""
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
    admissible_cohorts = tuple(
        int(value)
        for value in validation.cohort_diagnostics.loc[
            validation.cohort_diagnostics["admissible"],
            "subevent",
        ].tolist()
    )
    panel = keep_admissible_cohorts(data=panel, admissible_cohorts=admissible_cohorts)
    stacked_data = build_stacked_data(data=panel, config=config, validation=validation)

    cohort_params, model_summaries = estimate_cohort_models(
        stacked_data=stacked_data, config=config
    )
    joint_model = estimate_joint_stacked_model(stacked_data=stacked_data, config=config)
    covariance_by_event_time = extract_joint_covariance_by_event_time(
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
        covariance_by_event_time = scale_covariance_by_event_time(
            covariance_by_event_time=covariance_by_event_time,
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
        covariance_by_event_time=covariance_by_event_time,
    )

    if scale == "pre_birth":
        average_params["scale"] = "pre_birth"

    return StackedEventStudyResult(
        cohort_params=cohort_params,
        average_params=average_params,
        cohort_weights=cohort_weights,
        vcov_average=vcov_average,
        config=config,
        model_summaries=model_summaries,
        validation=validation,
        stacked_data=stacked_data if return_stacked_data else None,
    )


__all__ = ["estimate_stacked_eventstudy", "validate_stacked_eventstudy"]
