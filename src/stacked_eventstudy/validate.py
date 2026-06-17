"""Validation for stacked event-study estimation."""

from collections.abc import Sequence

import pandas as pd

from stacked_eventstudy.preprocess import prepare_panel_data
from stacked_eventstudy.types import EstimatorConfig, StackedEventStudyValidation
from stacked_eventstudy.utils import check_missing_columns, coerce_covariates

FEW_COHORT_WARNING_THRESHOLD = 2


def validate_stacked_eventstudy(
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
    heterogeneity_col: str | None = None,
    heterogeneity_weighting: str = "within",
) -> StackedEventStudyValidation:
    """Validate stacked event-study inputs and cohort feasibility.

    Args:
        data: Individual-level panel data.
        id_col: Column containing the individual identifier.
        age_col: Column containing age in integer years.
        treatment_age_col: Column containing age at first birth.
        outcome_col: Column containing the outcome variable.
        l_min: Minimum event time to validate.
        l_max: Maximum event time to validate.
        control_window: Number of future-treated cohorts used as controls.
        reference_event_time: Omitted event time in the event-study design.
        min_treatment_age: Optional lower bound on treated cohorts to estimate.
        max_treatment_age: Optional upper bound on treated cohorts to estimate.
        observed_min_age: Optional minimum observed age used in feasibility checks.
        calendar_year_col: Optional calendar-year column.
        covariates: Optional additional covariate columns.
        heterogeneity_col: Optional categorical column for group-specific effects.
        heterogeneity_weighting: Cohort weighting scheme for group-specific effects.

    Returns:
        A `StackedEventStudyValidation` object with errors, warnings, cohort-level
        diagnostics, and sample-level feasibility summaries.

    Raises:
        TypeError: If `data` is not a pandas DataFrame.
        ValueError: If the input has duplicate column names.
    """
    if not isinstance(data, pd.DataFrame):
        msg = "data must be a pandas DataFrame."
        raise TypeError(msg)
    if data.columns.duplicated().any():
        msg = "data contains duplicate column names."
        raise ValueError(msg)

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
        weights_col=None,
        cluster_col=None,
        heterogeneity_col=heterogeneity_col,
        heterogeneity_weighting=heterogeneity_weighting,
        scale="none",
        backend="statsmodels",
        covariance_policy="small_sample_correction",
        allow_unbalanced_treated_panel=True,
        return_stacked_data=False,
    )
    return _validate_with_config(data=data, config=config)


def _validate_with_config(  # noqa: C901, PLR0912
    data: pd.DataFrame,
    config: EstimatorConfig,
) -> StackedEventStudyValidation:
    """Validate the input data using a normalized configuration."""
    errors: list[str] = []
    warnings: list[str] = []

    required_columns = [
        config.id_col,
        config.age_col,
        config.treatment_age_col,
        config.outcome_col,
        *config.covariates,
    ]
    if config.calendar_year_col is not None:
        required_columns.append(config.calendar_year_col)
    if config.heterogeneity_col is not None:
        required_columns.append(config.heterogeneity_col)
    missing_columns = check_missing_columns(data, required_columns)
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}.")
    if (
        config.heterogeneity_col is not None
        and config.heterogeneity_col in config.covariates
    ):
        errors.append("heterogeneity_col must not also be listed in covariates.")

    if errors:
        return StackedEventStudyValidation(
            is_valid=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            observed_min_age=config.observed_min_age or -1,
            requested_cohort_range=None,
            admissible_cohort_range=None,
            cohort_diagnostics=pd.DataFrame(),
            sample_counts=_build_sample_counts(data, config),
            window_feasibility=_build_window_feasibility(
                config=config,
                observed_min_age_resolved=False,
                requested_cohort_range_nonempty=False,
            ),
        )

    panel = prepare_panel_data(data=data, config=config)

    missing_estimation_columns = _get_missing_estimation_columns(
        panel=panel,
        config=config,
    )
    if missing_estimation_columns:
        errors.append(
            "Estimation columns must be non-missing: "
            + ", ".join(
                f"{column} ({count} missing)"
                for column, count in missing_estimation_columns.items()
            )
            + ".",
        )

    if (
        panel.groupby("unit_id", sort=False)["treatment_age"]
        .nunique(dropna=False)
        .gt(1)
        .any()
    ):
        errors.append("Treatment age must be constant within individual.")

    if config.heterogeneity_col is not None and (
        panel.groupby("unit_id", sort=False)["heterogeneity_value"]
        .nunique(dropna=False)
        .gt(1)
        .any()
    ):
        errors.append("heterogeneity_col must be constant within individual.")

    if not _is_integer_like_series(panel["age"]):
        errors.append("Age must be recorded in integer years.")

    duplicate_id_age_count = int(panel.duplicated(subset=["unit_id", "age"]).sum())
    if duplicate_id_age_count > 0:
        errors.append("Duplicated id-age observations exist.")

    duplicate_id_calendar_year_count = 0
    if config.calendar_year_col is not None:
        duplicate_id_calendar_year_count = int(
            panel.duplicated(subset=["unit_id", "calendar_year"]).sum(),
        )
        if duplicate_id_calendar_year_count > 0:
            errors.append("Duplicated id-calendar_year observations exist.")

    if (
        config.reference_event_time < config.l_min
        or config.reference_event_time > config.l_max
    ):
        errors.append(
            "reference_event_time must lie inside the requested event-time window."
        )
    if config.l_max > config.control_window - 1:
        errors.append("l_max must be less than or equal to control_window - 1.")
    if config.l_min >= config.l_max:
        errors.append("l_min must be strictly smaller than l_max.")
    if config.backend not in {"statsmodels", "pyfixest"}:
        errors.append("backend must be either 'statsmodels' or 'pyfixest'.")
    if config.covariance_policy not in {"small_sample_correction", "none"}:
        errors.append(
            "covariance_policy must be either 'small_sample_correction' or 'none'."
        )
    if not isinstance(config.allow_unbalanced_treated_panel, bool):
        errors.append("allow_unbalanced_treated_panel must be a boolean.")
    if config.heterogeneity_weighting not in {"within", "overall"}:
        errors.append("heterogeneity_weighting must be either 'within' or 'overall'.")

    resolved_observed_min_age = (
        int(panel["age"].min())
        if config.observed_min_age is None
        else config.observed_min_age
    )
    max_observed_treatment_age = int(panel["treatment_age"].max())
    requested_min_cohort = max(
        resolved_observed_min_age - config.l_min,
        config.min_treatment_age if config.min_treatment_age is not None else -(10**9),
    )
    requested_max_cohort = min(
        max_observed_treatment_age - config.control_window,
        config.max_treatment_age if config.max_treatment_age is not None else 10**9,
    )
    requested_range = None
    if requested_min_cohort <= requested_max_cohort:
        requested_range = (requested_min_cohort, requested_max_cohort)
    else:
        errors.append("The requested cohort range is empty.")

    cohort_diagnostics = _diagnose_cohorts(
        panel=panel,
        config=config,
        requested_range=requested_range,
    )
    admissible_rows = cohort_diagnostics.loc[cohort_diagnostics["admissible"]]
    admissible_range = None
    if not admissible_rows.empty:
        admissible_range = (
            int(admissible_rows["subevent"].min()),
            int(admissible_rows["subevent"].max()),
        )
    else:
        errors.append(
            "No admissible treated cohorts remain after applying restrictions."
        )

    if (
        int(admissible_rows.shape[0]) <= FEW_COHORT_WARNING_THRESHOLD
        and int(admissible_rows.shape[0]) > 0
    ):
        warnings.append("Very few admissible cohorts remain.")

    sample_counts = _build_sample_counts(
        data=panel,
        config=config,
        duplicate_id_age_count=duplicate_id_age_count,
        duplicate_id_calendar_year_count=duplicate_id_calendar_year_count,
    )
    window_feasibility = _build_window_feasibility(
        config=config,
        observed_min_age_resolved=True,
        requested_cohort_range_nonempty=requested_range is not None,
    )

    return StackedEventStudyValidation(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        observed_min_age=resolved_observed_min_age,
        requested_cohort_range=requested_range,
        admissible_cohort_range=admissible_range,
        cohort_diagnostics=cohort_diagnostics,
        sample_counts=sample_counts,
        window_feasibility=window_feasibility,
    )


def _build_sample_counts(
    data: pd.DataFrame,
    config: EstimatorConfig,
    duplicate_id_age_count: int = 0,
    duplicate_id_calendar_year_count: int = 0,
) -> pd.DataFrame:
    """Build sample-count diagnostics."""
    id_column = config.id_col if config.id_col in data.columns else "unit_id"
    treatment_column = (
        config.treatment_age_col
        if config.treatment_age_col in data.columns
        else "treatment_age"
    )
    rows = [
        {"metric": "n_rows", "value": int(data.shape[0])},
        {"metric": "n_individuals", "value": int(data[id_column].nunique())},
        {
            "metric": "n_unique_treatment_ages",
            "value": int(data[treatment_column].nunique()),
        },
        {
            "metric": "n_missing_treatment_age",
            "value": int(data[treatment_column].isna().sum()),
        },
        {"metric": "n_duplicate_id_age_rows", "value": duplicate_id_age_count},
        {
            "metric": "n_duplicate_id_calendar_year_rows",
            "value": duplicate_id_calendar_year_count,
        },
    ]
    return pd.DataFrame(rows)


def _build_window_feasibility(
    config: EstimatorConfig,
    observed_min_age_resolved: bool,  # noqa: FBT001
    requested_cohort_range_nonempty: bool,  # noqa: FBT001
) -> pd.DataFrame:
    """Build event-window feasibility diagnostics."""
    rows = [
        {
            "rule": "reference_event_time_in_window",
            "passed": config.l_min <= config.reference_event_time <= config.l_max,
            "value": str(config.reference_event_time),
        },
        {
            "rule": "l_max_leq_control_window_minus_1",
            "passed": config.l_max <= config.control_window - 1,
            "value": f"{config.l_max} <= {config.control_window - 1}",
        },
        {
            "rule": "l_min_lt_l_max",
            "passed": config.l_min < config.l_max,
            "value": f"{config.l_min} < {config.l_max}",
        },
        {
            "rule": "observed_min_age_resolved",
            "passed": observed_min_age_resolved,
            "value": str(config.observed_min_age),
        },
        {
            "rule": "requested_cohort_range_nonempty",
            "passed": requested_cohort_range_nonempty,
            "value": "",
        },
    ]
    return pd.DataFrame(rows)


def _diagnose_cohorts(
    panel: pd.DataFrame,
    config: EstimatorConfig,
    requested_range: tuple[int, int] | None,
) -> pd.DataFrame:
    """Create cohort-level diagnostics."""
    unique_cohorts = sorted(
        int(value) for value in panel["treatment_age"].dropna().unique()
    )
    requested_cohorts: set[int] = set()
    if requested_range is not None:
        requested_cohorts = set(range(requested_range[0], requested_range[1] + 1))

    rows: list[dict[str, object]] = []
    required_event_times = set(range(config.l_min, config.l_max + 1))
    required_treated_event_times = required_event_times
    required_control_event_times = set(range(config.l_min, min(-1, config.l_max) + 1))

    for cohort in unique_cohorts:
        in_requested_range = cohort in requested_cohorts
        treated = panel.loc[panel["treatment_age"] == cohort].copy()
        treated["event_time"] = treated["age"] - cohort
        treated = treated.loc[
            (treated["event_time"] >= config.l_min)
            & (treated["event_time"] <= config.l_max)
        ]

        controls = panel.loc[
            panel["treatment_age"].between(cohort + 1, cohort + config.control_window),
        ].copy()
        controls = controls.loc[controls["age"] < controls["treatment_age"]]
        controls["event_time"] = controls["age"] - cohort
        controls = controls.loc[
            (controls["event_time"] >= config.l_min)
            & (controls["event_time"] <= config.l_max)
        ]

        treated_coverage = {int(value) for value in treated["event_time"].unique()}
        control_coverage = {int(value) for value in controls["event_time"].unique()}
        has_rolling_controls = not controls.empty
        treated_complete = required_treated_event_times.issubset(treated_coverage)
        controls_complete = required_control_event_times.issubset(control_coverage)
        admissible = (
            in_requested_range
            and has_rolling_controls
            and treated_complete
            and controls_complete
        )

        drop_reason = ""
        if not in_requested_range:
            drop_reason = "outside_requested_range"
        elif not has_rolling_controls:
            drop_reason = "no_rolling_window_controls"
        elif not treated_complete:
            drop_reason = "missing_treated_event_times"
        elif not controls_complete:
            drop_reason = "missing_control_event_times"

        rows.append(
            {
                "subevent": cohort,
                "requested": in_requested_range,
                "in_requested_range": in_requested_range,
                "has_rolling_controls": has_rolling_controls,
                "treated_complete": treated_complete,
                "controls_complete": controls_complete,
                "admissible": admissible,
                "drop_reason": drop_reason,
                "n_treated_individuals": int(treated["unit_id"].nunique()),
                "n_control_individuals": int(controls["unit_id"].nunique()),
                "n_treated_obs": int(treated.shape[0]),
                "n_control_obs": int(controls.shape[0]),
                "treated_event_time_min": treated["event_time"].min()
                if not treated.empty
                else pd.NA,
                "treated_event_time_max": treated["event_time"].max()
                if not treated.empty
                else pd.NA,
                "control_event_time_min": controls["event_time"].min()
                if not controls.empty
                else pd.NA,
                "control_event_time_max": controls["event_time"].max()
                if not controls.empty
                else pd.NA,
            },
        )

    return pd.DataFrame(rows).sort_values("subevent").reset_index(drop=True)


def _get_missing_estimation_columns(
    panel: pd.DataFrame,
    config: EstimatorConfig,
) -> dict[str, int]:
    """Return missing-value counts for columns required during estimation."""
    estimation_columns = [
        "unit_id",
        "age",
        "treatment_age",
        "outcome",
        *config.covariates,
        "input_weight",
        "cluster_id",
    ]
    if config.heterogeneity_col is not None:
        estimation_columns.append("heterogeneity_value")
    missing_counts = panel.loc[:, estimation_columns].isna().sum()
    return {
        column: int(count) for column, count in missing_counts.items() if int(count) > 0
    }


def _is_integer_like_series(series: pd.Series) -> bool:
    """Return whether a series is integer-like."""
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.isna().any():
        return False
    return ((numeric_values % 1) == 0).all()
