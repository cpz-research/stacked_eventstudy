"""Estimation helpers."""

import pandas as pd

from stacked_eventstudy.types import EstimatorConfig
from stacked_eventstudy.utils import make_confidence_interval


def estimate_cohort_models(
    stacked_data: pd.DataFrame,
    config: EstimatorConfig,
) -> tuple[pd.DataFrame, dict[int, object]]:
    """Estimate separate cohort-specific regressions."""
    summaries: dict[int, object] = {}
    results: list[pd.DataFrame] = []
    for subevent in sorted(int(value) for value in stacked_data["subevent"].unique()):
        subevent_data = stacked_data.loc[stacked_data["subevent"] == subevent].copy()
        fitted_model = fit_subevent_model(subevent_data=subevent_data, config=config)
        summaries[subevent] = fitted_model
        results.append(
            extract_subevent_params(
                fitted_model=fitted_model, subevent=subevent, config=config
            )
        )
    cohort_params = pd.concat(results, ignore_index=True)
    return cohort_params.sort_values(["subevent", "event_time"]).reset_index(
        drop=True
    ), summaries


def estimate_joint_stacked_model(
    stacked_data: pd.DataFrame,
    config: EstimatorConfig,
) -> object:
    """Estimate the fully interacted stacked regression."""
    return fit_joint_model(stacked_data=stacked_data, config=config)


def fit_subevent_model(subevent_data: pd.DataFrame, config: EstimatorConfig) -> object:
    """Fit one cohort-specific regression model."""
    design_data = _add_subevent_regressors(data=subevent_data, config=config)
    formula = _make_subevent_formula(config=config)
    return _fit_formula_model(data=design_data, formula=formula, config=config)


def fit_joint_model(stacked_data: pd.DataFrame, config: EstimatorConfig) -> object:
    """Fit the joint stacked regression model."""
    design_data = _add_joint_regressors(data=stacked_data, config=config)
    formula = _make_joint_formula(data=design_data, config=config)
    return _fit_formula_model(data=design_data, formula=formula, config=config)


def extract_subevent_params(
    fitted_model: object,
    subevent: int,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Extract event-time coefficients from a cohort-specific model."""
    rows: list[dict[str, object]] = []
    for event_time in _non_reference_event_times(config):
        term_label = _subevent_regressor_name(event_time)
        if term_label not in fitted_model.params.index:
            continue
        estimate = float(fitted_model.params.loc[term_label])
        std_error = float(fitted_model.bse.loc[term_label])
        ci_low, ci_high = make_confidence_interval(
            estimate=estimate, std_error=std_error
        )
        rows.append(
            {
                "subevent": subevent,
                "event_time": event_time,
                "estimate": estimate,
                "std_error": std_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "scale": "none",
            },
        )
    return pd.DataFrame(rows)


def extract_joint_covariance_by_event_time(
    fitted_model: object,
    cohort_params: pd.DataFrame,
    config: EstimatorConfig,
) -> dict[int, pd.DataFrame]:
    """Extract covariance blocks by event time from the joint model."""
    covariance_matrix = fitted_model.cov_params()
    covariance_by_event_time: dict[int, pd.DataFrame] = {}
    subevents_by_event_time = (
        cohort_params.groupby("event_time", sort=True)["subevent"].apply(list).to_dict()
    )
    for event_time, subevents in subevents_by_event_time.items():
        labels = [
            _joint_regressor_name(subevent=int(subevent), event_time=int(event_time))
            for subevent in subevents
        ]
        covariance_block = covariance_matrix.loc[labels, labels].copy()
        covariance_block.index = [int(subevent) for subevent in subevents]
        covariance_block.columns = [int(subevent) for subevent in subevents]
        covariance_by_event_time[int(event_time)] = covariance_block
    return covariance_by_event_time


def _fit_formula_model(
    data: pd.DataFrame, formula: str, config: EstimatorConfig
) -> object:
    """Fit a formula-based regression with clustered standard errors."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as error:
        msg = "statsmodels is required for estimation."
        raise ImportError(msg) from error

    weights = data["input_weight"] if config.weights_col is not None else None
    fit_kwargs = {
        "cov_type": "cluster",
        "cov_kwds": {"groups": data["cluster_id"], "use_correction": False},
    }

    if weights is None:
        model = smf.ols(formula=formula, data=data)
    else:
        model = smf.wls(formula=formula, data=data, weights=weights)
    return model.fit(**fit_kwargs)


def _make_subevent_formula(config: EstimatorConfig) -> str:
    """Create the cohort-specific formula."""
    regressor_terms = [
        _subevent_regressor_name(event_time)
        for event_time in _non_reference_event_times(config)
    ]
    base_terms = [*regressor_terms, "C(unit_id)", "C(age)", *config.covariates]
    return "outcome ~ 0 + " + " + ".join(base_terms)


def _make_joint_formula(data: pd.DataFrame, config: EstimatorConfig) -> str:
    """Create the joint stacked formula."""
    regressor_terms = [
        column
        for column in data.columns
        if column.startswith("coef_s") and "_l" in column
    ]
    base_terms = [
        *regressor_terms,
        "C(unit_subevent_id)",
        "C(subevent):C(age)",
        *config.covariates,
    ]
    return "outcome ~ 0 + " + " + ".join(base_terms)


def _add_subevent_regressors(
    data: pd.DataFrame, config: EstimatorConfig
) -> pd.DataFrame:
    """Add explicit event-time indicators to one subevent stack."""
    design_data = data.copy()
    for event_time in _non_reference_event_times(config):
        column_name = _subevent_regressor_name(event_time)
        design_data[column_name] = (
            (design_data["event_time"] == event_time)
            & (design_data["treated_in_subevent"] == 1)
        ).astype(int)
    return design_data


def _add_joint_regressors(data: pd.DataFrame, config: EstimatorConfig) -> pd.DataFrame:
    """Add explicit cohort-by-event-time indicators to the full stack."""
    design_data = data.copy()
    cohort_event_pairs = (
        design_data.loc[
            design_data["treated_in_subevent"] == 1, ["subevent", "event_time"]
        ]
        .drop_duplicates()
        .sort_values(["subevent", "event_time"])
        .itertuples(index=False, name=None)
    )
    for subevent, event_time in cohort_event_pairs:
        if int(event_time) == config.reference_event_time:
            continue
        column_name = _joint_regressor_name(
            subevent=int(subevent), event_time=int(event_time)
        )
        design_data[column_name] = (
            (design_data["subevent"] == int(subevent))
            & (design_data["event_time"] == int(event_time))
            & (design_data["treated_in_subevent"] == 1)
        ).astype(int)
    return design_data


def _non_reference_event_times(config: EstimatorConfig) -> tuple[int, ...]:
    """Return event times excluding the omitted reference period."""
    return tuple(
        event_time
        for event_time in range(config.l_min, config.l_max + 1)
        if event_time != config.reference_event_time
    )


def _encode_event_time(event_time: int) -> str:
    """Encode an event time for use in a valid column name."""
    prefix = "m" if event_time < 0 else "p"
    return f"{prefix}{abs(event_time)}"


def _subevent_regressor_name(event_time: int) -> str:
    """Return a cohort-model regressor name."""
    return f"event_{_encode_event_time(event_time)}"


def _joint_regressor_name(subevent: int, event_time: int) -> str:
    """Return a joint-model regressor name."""
    return f"coef_s{subevent}_l{_encode_event_time(event_time)}"
