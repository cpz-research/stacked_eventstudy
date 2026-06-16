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
    if config.heterogeneity_col is not None:
        return _extract_heterogeneous_cohort_params_from_joint_model(
            fitted_model=fitted_model,
            stacked_data=stacked_data,
            config=config,
        )

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
    if config.heterogeneity_col is not None:
        parameter_index = pd.MultiIndex.from_frame(
            cohort_params.loc[:, ["heterogeneity_value", "event_time", "subevent"]],
            names=["heterogeneity_value", "event_time", "subevent"],
        )
        labels = cohort_params["term_label"].tolist()
        joint_covariance = covariance_matrix.loc[labels, labels].copy()
        joint_covariance.index = parameter_index
        joint_covariance.columns = parameter_index
        return joint_covariance

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
    use_correction = config.covariance_policy == "stata"
    fit_kwargs = {
        "cov_type": "cluster",
        "cov_kwds": {"groups": data["cluster_id"], "use_correction": use_correction},
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
    if config.covariance_policy == "stata":
        small_sample_correction = pf.ssc()
    else:
        small_sample_correction = pf.ssc(k_adj=False, G_adj=False)
    return pf.feols(
        formula,
        data=data,
        vcov={"CRV1": "cluster_id"},
        weights=weights,
        ssc=small_sample_correction,
        fixef_rm="none",
    )


def _make_statsmodels_joint_formula(data: pd.DataFrame, config: EstimatorConfig) -> str:
    """Create the statsmodels joint stacked formula."""
    regressor_terms = _get_joint_regressor_terms(data=data)
    fixed_effect_terms = ["C(unit_subevent_id)", "C(subevent):C(age)"]
    if config.heterogeneity_col is not None:
        fixed_effect_terms = [
            "C(unit_subevent_id)",
            "C(subevent_age_heterogeneity_id)",
        ]
    base_terms = [
        *regressor_terms,
        *fixed_effect_terms,
        *config.covariates,
    ]
    return "outcome ~ 0 + " + " + ".join(base_terms)


def _make_pyfixest_joint_formula(data: pd.DataFrame, config: EstimatorConfig) -> str:
    """Create the pyfixest joint stacked formula."""
    regressor_terms = _get_joint_regressor_terms(data=data)
    right_hand_side = " + ".join([*regressor_terms, *config.covariates])
    fixed_effect_terms = "unit_subevent_id + subevent^age"
    if config.heterogeneity_col is not None:
        fixed_effect_terms = "unit_subevent_id + subevent_age_heterogeneity_id"
    return f"outcome ~ 0 + {right_hand_side} | {fixed_effect_terms}"


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
    if config.heterogeneity_col is not None:
        design_data["subevent_age_heterogeneity_id"] = (
            design_data["subevent"].astype(str)
            + "__"
            + design_data["age"].astype(str)
            + "__"
            + design_data["heterogeneity_value"].astype(str)
        )
        return _add_heterogeneous_joint_regressors(data=design_data, config=config)

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


def _extract_heterogeneous_cohort_params_from_joint_model(
    fitted_model: object,
    stacked_data: pd.DataFrame,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Extract group-specific event-time coefficients from the joint model."""
    rows: list[dict[str, object]] = []
    value_codes = _get_heterogeneity_value_codes(data=stacked_data)
    cohort_event_pairs = (
        stacked_data.loc[
            stacked_data["treated_in_subevent"] == 1,
            ["heterogeneity_value", "subevent", "event_time"],
        ]
        .drop_duplicates()
        .sort_values(["heterogeneity_value", "subevent", "event_time"])
        .itertuples(index=False, name=None)
    )
    params = _get_model_params(fitted_model=fitted_model, config=config)
    standard_errors = _get_model_standard_errors(
        fitted_model=fitted_model,
        config=config,
    )
    for heterogeneity_value, subevent, event_time in cohort_event_pairs:
        if int(event_time) == config.reference_event_time:
            continue
        term_label = _heterogeneous_joint_regressor_name(
            heterogeneity_code=value_codes[heterogeneity_value],
            subevent=int(subevent),
            event_time=int(event_time),
        )
        if term_label not in params.index:
            continue
        estimate = float(params.loc[term_label])
        std_error = float(standard_errors.loc[term_label])
        ci_low, ci_high = make_confidence_interval(
            estimate=estimate,
            std_error=std_error,
        )
        rows.append(
            {
                "heterogeneity_col": config.heterogeneity_col,
                "heterogeneity_value": heterogeneity_value,
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
        .sort_values(["heterogeneity_value", "subevent", "event_time"])
        .reset_index(drop=True)
    )


def _add_heterogeneous_joint_regressors(
    data: pd.DataFrame,
    config: EstimatorConfig,
) -> pd.DataFrame:
    """Add group-specific cohort-by-event-time indicators to the full stack."""
    design_data = data.copy()
    value_codes = _get_heterogeneity_value_codes(data=design_data)
    cohort_event_pairs = (
        design_data.loc[
            design_data["treated_in_subevent"] == 1,
            ["heterogeneity_value", "subevent", "event_time"],
        ]
        .drop_duplicates()
        .sort_values(["heterogeneity_value", "subevent", "event_time"])
        .itertuples(index=False, name=None)
    )
    for heterogeneity_value, subevent, event_time in cohort_event_pairs:
        if int(event_time) == config.reference_event_time:
            continue
        column_name = _heterogeneous_joint_regressor_name(
            heterogeneity_code=value_codes[heterogeneity_value],
            subevent=int(subevent),
            event_time=int(event_time),
        )
        design_data[column_name] = (
            (design_data["heterogeneity_value"] == heterogeneity_value)
            & (design_data["subevent"] == int(subevent))
            & (design_data["event_time"] == int(event_time))
            & (design_data["treated_in_subevent"] == 1)
        ).astype(int)
    return design_data


def _get_joint_regressor_terms(data: pd.DataFrame) -> list[str]:
    """Return generated joint-regressor columns."""
    return [
        column
        for column in data.columns
        if column.startswith(("coef_s", "coef_h")) and "_l" in column
    ]


def _get_heterogeneity_value_codes(data: pd.DataFrame) -> dict[object, int]:
    """Return deterministic integer codes for heterogeneity values."""
    values = data["heterogeneity_value"].drop_duplicates().tolist()
    sorted_values = sorted(values, key=lambda value: (str(type(value)), str(value)))
    return {value: index for index, value in enumerate(sorted_values)}


def _heterogeneous_joint_regressor_name(
    heterogeneity_code: int,
    subevent: int,
    event_time: int,
) -> str:
    """Return a group-specific joint-model regressor name."""
    return f"coef_h{heterogeneity_code}_s{subevent}_l{_encode_event_time(event_time)}"
