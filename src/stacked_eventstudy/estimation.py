"""Estimation helpers."""

import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf

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
    if config.backend == "statsmodels":
        formula = _make_statsmodels_joint_formula(data=design_data, config=config)
        return _fit_statsmodels_formula_model(
            data=design_data,
            formula=formula,
            config=config,
        )
    if config.backend == "pyfixest":
        formula = _make_pyfixest_joint_formula(data=design_data, config=config)
        return _fit_pyfixest_formula_model(
            data=design_data,
            formula=formula,
            config=config,
        )
    msg = f"Unsupported backend: {config.backend}."
    raise ValueError(msg)


def extract_cohort_params_from_joint_model(
    fitted_model: object,
    stacked_data: pd.DataFrame,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Extract cohort-specific event-time coefficients from the joint model."""
    rows: list[dict[str, object]] = []
    cohort_event_pairs = (
        stacked_data.loc[
            stacked_data["treated_in_subevent"] == 1, ["subevent", "event_time"]
        ]
        .drop_duplicates()
        .sort_values(["subevent", "event_time"])
        .itertuples(index=False, name=None)
    )
    params = _get_model_params(fitted_model=fitted_model, config=config)
    standard_errors = _get_model_standard_errors(
        fitted_model=fitted_model, config=config
    )
    for subevent, event_time in cohort_event_pairs:
        if int(event_time) == config.reference_event_time:
            continue
        term_label = _joint_regressor_name(
            subevent=int(subevent), event_time=int(event_time)
        )
        if term_label not in params.index:
            continue
        estimate = float(params.loc[term_label])
        std_error = float(standard_errors.loc[term_label])
        ci_low, ci_high = make_confidence_interval(
            estimate=estimate, std_error=std_error
        )
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
    return (
        pd.DataFrame(rows)
        .sort_values(["subevent", "event_time"])
        .reset_index(drop=True)
    )


def extract_joint_parameter_covariance(
    fitted_model: object,
    cohort_params: pd.DataFrame,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Extract the joint covariance matrix for cohort-event coefficients."""
    covariance_matrix = _get_model_covariance(fitted_model=fitted_model, config=config)
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


def _fit_statsmodels_formula_model(
    data: pd.DataFrame, formula: str, config: EstimatorConfig
) -> object:
    """Fit a statsmodels formula regression with clustered standard errors."""
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


def _fit_pyfixest_formula_model(
    data: pd.DataFrame, formula: str, config: EstimatorConfig
) -> object:
    """Fit a pyfixest formula regression with clustered standard errors."""
    weights = "input_weight" if config.weights_col is not None else None
    return pf.feols(
        formula,
        data=data,
        vcov={"CRV1": "cluster_id"},
        weights=weights,
        ssc=pf.ssc(k_adj=False, G_adj=False),
        fixef_rm="none",
    )


def _make_statsmodels_joint_formula(data: pd.DataFrame, config: EstimatorConfig) -> str:
    """Create the statsmodels joint stacked formula."""
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


def _make_pyfixest_joint_formula(data: pd.DataFrame, config: EstimatorConfig) -> str:
    """Create the pyfixest joint stacked formula."""
    regressor_terms = [
        column
        for column in data.columns
        if column.startswith("coef_s") and "_l" in column
    ]
    right_hand_side = " + ".join([*regressor_terms, *config.covariates])
    return f"outcome ~ 0 + {right_hand_side} | unit_subevent_id + subevent^age"


def _get_model_params(fitted_model: object, config: EstimatorConfig) -> pd.Series:
    """Return coefficient estimates from a fitted backend model."""
    if config.backend == "statsmodels":
        return fitted_model.params
    if config.backend == "pyfixest":
        return fitted_model.coef()
    msg = f"Unsupported backend: {config.backend}."
    raise ValueError(msg)


def _get_model_standard_errors(
    fitted_model: object, config: EstimatorConfig
) -> pd.Series:
    """Return coefficient standard errors from a fitted backend model."""
    if config.backend == "statsmodels":
        return fitted_model.bse
    if config.backend == "pyfixest":
        return fitted_model.se()
    msg = f"Unsupported backend: {config.backend}."
    raise ValueError(msg)


def _get_model_covariance(
    fitted_model: object,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Return the coefficient covariance matrix from a fitted backend model."""
    if config.backend == "statsmodels":
        return fitted_model.cov_params()
    if config.backend == "pyfixest":
        parameter_index = fitted_model.coef().index
        # pyfixest exposes SEs publicly, but the full covariance matrix is stored here.
        return pd.DataFrame(
            fitted_model._vcov,  # noqa: SLF001
            index=parameter_index,
            columns=parameter_index,
        )
    msg = f"Unsupported backend: {config.backend}."
    raise ValueError(msg)


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


def _encode_event_time(event_time: int) -> str:
    """Encode an event time for use in a valid column name."""
    prefix = "m" if event_time < 0 else "p"
    return f"{prefix}{abs(event_time)}"


def _joint_regressor_name(subevent: int, event_time: int) -> str:
    """Return a joint-model regressor name."""
    return f"coef_s{subevent}_l{_encode_event_time(event_time)}"
