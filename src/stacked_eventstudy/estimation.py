"""Estimation helpers."""

import pandas as pd

from stacked_eventstudy.types import EstimatorConfig
from stacked_eventstudy.utils import make_confidence_interval


def estimate_joint_stacked_model(
    stacked_data: pd.DataFrame,
    config: EstimatorConfig,
) -> object:
    """Estimate the fully interacted stacked regression."""
    return fit_joint_model(stacked_data=stacked_data, config=config)


def fit_joint_model(stacked_data: pd.DataFrame, config: EstimatorConfig) -> object:
    """Fit the joint stacked regression model."""
    design_data = _add_joint_regressors(data=stacked_data, config=config)
    formula = _make_joint_formula(data=design_data, config=config)
    return _fit_formula_model(data=design_data, formula=formula, config=config)


def extract_cohort_params_from_joint_model(
    fitted_model: object,
    stacked_data: pd.DataFrame,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Extract cohort-specific event-time coefficients from the joint model."""
    rows: list[dict[str, object]] = []
    cohort_event_pairs = (
        stacked_data.loc[stacked_data["treated_in_subevent"] == 1, ["subevent", "event_time"]]
        .drop_duplicates()
        .sort_values(["subevent", "event_time"])
        .itertuples(index=False, name=None)
    )
    for subevent, event_time in cohort_event_pairs:
        if int(event_time) == config.reference_event_time:
            continue
        term_label = _joint_regressor_name(subevent=int(subevent), event_time=int(event_time))
        if term_label not in fitted_model.params.index:
            continue
        estimate = float(fitted_model.params.loc[term_label])
        std_error = float(fitted_model.bse.loc[term_label])
        ci_low, ci_high = make_confidence_interval(estimate=estimate, std_error=std_error)
        rows.append(
            {
                "subevent": int(subevent),
                "event_time": int(event_time),
                "term_label": term_label,
                "estimate": estimate,
                "std_error": std_error,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "scale": "none",
            },
        )
    return pd.DataFrame(rows).sort_values(["subevent", "event_time"]).reset_index(drop=True)


def extract_joint_parameter_covariance(
    fitted_model: object,
    cohort_params: pd.DataFrame,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Extract the joint covariance matrix for cohort-event coefficients."""
    covariance_matrix = fitted_model.cov_params()
    parameter_index = pd.MultiIndex.from_frame(
        cohort_params.loc[:, ["event_time", "subevent"]],
        names=["event_time", "subevent"],
    )
    labels = [
        _joint_regressor_name(subevent=int(subevent), event_time=int(event_time))
        for event_time, subevent in parameter_index.tolist()
    ]
    joint_covariance = covariance_matrix.loc[labels, labels].copy()
    joint_covariance.index = parameter_index
    joint_covariance.columns = parameter_index
    return joint_covariance


def _fit_formula_model(data: pd.DataFrame, formula: str, config: EstimatorConfig) -> object:
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


def _make_joint_formula(data: pd.DataFrame, config: EstimatorConfig) -> str:
    """Create the joint stacked formula."""
    regressor_terms = [
        column
        for column in data.columns
        if column.startswith("coef_s") and "_l" in column
    ]
    base_terms = [*regressor_terms, "C(unit_subevent_id)", "C(subevent):C(age)", *config.covariates]
    return "outcome ~ 0 + " + " + ".join(base_terms)


def _add_joint_regressors(data: pd.DataFrame, config: EstimatorConfig) -> pd.DataFrame:
    """Add explicit cohort-by-event-time indicators to the full stack."""
    design_data = data.copy()
    cohort_event_pairs = (
        design_data.loc[design_data["treated_in_subevent"] == 1, ["subevent", "event_time"]]
        .drop_duplicates()
        .sort_values(["subevent", "event_time"])
        .itertuples(index=False, name=None)
    )
    for subevent, event_time in cohort_event_pairs:
        if int(event_time) == config.reference_event_time:
            continue
        column_name = _joint_regressor_name(subevent=int(subevent), event_time=int(event_time))
        design_data[column_name] = (
            (design_data["subevent"] == int(subevent))
            & (design_data["event_time"] == int(event_time))
            & (design_data["treated_in_subevent"] == 1)
        ).astype(int)
    return design_data


def _encode_event_time(event_time: int) -> str:
    """Encode an event time for use in a valid column name."""
    prefix = "m" if event_time < 0 else "p"
    return f"{prefix}{abs(event_time)}"


def _joint_regressor_name(subevent: int, event_time: int) -> str:
    """Return a joint-model regressor name."""
    return f"coef_s{subevent}_l{_encode_event_time(event_time)}"
