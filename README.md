# stacked_eventstudy

`stacked_eventstudy` implements a stacked difference-in-differences estimator with
rolling-window controls by age at first birth.

The current package is built around the heterogeneity-robust stacked estimator described
in Melentyeva and Riedel (2025), where each treatment-age cohort is estimated in its own
stacked subevent and the resulting event-study coefficients are aggregated using focal
cohort observation shares.

The package is validated against the authors' Stata implementation available at
<https://gitlab.com/lukasriedel/MelentyevaRiedel_StackedDiD>.

## Status

The package currently includes:

- public APIs for validation and estimation
- validation of panel structure, treatment-age consistency, and cohort feasibility
- stacked data construction with rolling-window controls
- cohort-specific and aggregated event-study outputs
- clustered standard errors from a joint stacked regression
- optional pre-birth scaling
- synthetic tests for validation, aggregation, and recovery in simple designs

## Installation

This repository uses `pixi` for environment management.

```bash
pixi install
```

Run tests with:

```bash
pixi run pytest
```

## Data requirements

Input data must be an individual-level panel with:

- an individual identifier
- age in integer years
- age at first birth for every individual
- an outcome variable

Optional inputs:

- calendar year
- covariates
- weights
- a custom clustering variable

The current implementation assumes:

- all individuals are eventually treated
- age is observed in integer years
- treatment age is constant within individual
- there are no duplicate `id-age` observations
- if `calendar_year_col` is supplied, there are no duplicate `id-calendar_year`
  observations

## Estimator design

Let:

- `A_i` be individual `i`'s age at first birth
- `a` be observed age
- `l = a - A_i` be event time

For each treated cohort `A`:

- treated observations come from individuals with `A_i == A`
- control observations come from individuals with `A_i in {A + 1, ..., A + G}`, where
  `G = control_window`
- controls are restricted to pre-birth observations, `age < treatment_age`
- both treated and controls are aligned relative to the treated cohort's birth age `A`

That last point matters:

- controls are not aligned to their own birth age
- in subevent `A`, both groups use `event_time = age - A`

The estimator then:

1. runs a cohort-specific regression for each admissible treatment-age cohort
1. estimates a joint stacked model for covariance extraction
1. aggregates cohort-specific coefficients using focal treated cohort observation shares

## Paper-to-code traceability

The table below maps the main implementation claims from Section IV of Melentyeva and
Riedel (2025) to the package modules and behavioral tests that guard them.

| Paper component                                                                                                                                           | Implementation                                                                                                                    | Test coverage                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rolling-window controls use future-treated cohorts and exclude already-treated control observations.                                                      | `src/stacked_eventstudy/stacking.py::build_subevent_stack`                                                                        | `tests/test_paper_alignment.py::test_clean_room_reference_matches_package_cohort_estimates`; `tests/test_paper_alignment.py::test_stacked_data_satisfies_paper_invariants`                    |
| Subevent-specific event-study indicators are estimated with subevent-by-age fixed effects, unit-by-subevent fixed effects, and clustered standard errors. | `src/stacked_eventstudy/estimation.py::fit_joint_model`                                                                           | `tests/test_paper_alignment.py::test_clean_room_reference_matches_package_cohort_estimates`; `tests/test_estimation_properties.py::test_pyfixest_backend_matches_statsmodels_average_effects` |
| The requested event-time window must be compatible with the rolling control window and feasible cohort range.                                             | `src/stacked_eventstudy/validate.py::_validate_with_config`                                                                       | `tests/test_validation.py::test_validate_rejects_infeasible_window`; `tests/test_paper_alignment.py::test_stacked_data_satisfies_paper_invariants`                                            |
| Age-at-birth-specific estimates are aggregated using cohort sample shares and the joint covariance matrix.                                                | `src/stacked_eventstudy/aggregation.py::compute_cohort_weights`; `src/stacked_eventstudy/aggregation.py::aggregate_cohort_params` | `tests/test_estimation_properties.py::test_aggregation_matches_weighted_cohort_average`; `tests/test_paper_alignment.py::test_cohort_weights_use_focal_observation_mass`                      |
| Pre-birth scaling divides cohort-specific effects and covariances by cohort-specific pre-birth levels.                                                    | `src/stacked_eventstudy/scaling.py::compute_pre_birth_levels`; `src/stacked_eventstudy/scaling.py::scale_cohort_params`           | `tests/test_estimation_properties.py::test_pre_birth_scaling_matches_manual_scaling`                                                                                                          |

## Main functions

### `validate_stacked_eventstudy(...)`

Use this first when you want to inspect whether the requested event-time window and
rolling control design are feasible in your sample.

It returns a `StackedEventStudyValidation` object with:

- `is_valid`
- `errors`
- `warnings`
- `cohort_diagnostics`
- `sample_counts`
- `window_feasibility`

### `estimate_stacked_eventstudy(...)`

Main estimator signature:

```python
estimate_stacked_eventstudy(
    data,
    id_col,
    age_col,
    treatment_age_col,
    outcome_col,
    l_min=-3,
    l_max=4,
    control_window=5,
    reference_event_time=-1,
    min_treatment_age=None,
    max_treatment_age=None,
    observed_min_age=None,
    calendar_year_col=None,
    covariates=(),
    weights_col=None,
    cluster_col=None,
    heterogeneity_col=None,
    heterogeneity_weighting="within",
    scale="none",
    backend="statsmodels",
    covariance_policy="stata",
    return_stacked_data=False,
)
```

Important arguments:

- `control_window`: width of the future-treated control window
- `reference_event_time`: omitted event time, usually `-1`
- `backend`: regression backend, either `"statsmodels"` or `"pyfixest"`
- `heterogeneity_col`: optional categorical, time-invariant column for group-specific
  effects
- `heterogeneity_weighting`: group-specific aggregation weights, either `"within"` or
  `"overall"`
- `scale="pre_birth"`: rescales effects by the treated cohort's mean outcome at the
  reference period
- `cluster_col`: overrides default clustering on the original individual id
- `covariance_policy`: clustered covariance convention, either `"stata"` for Stata-like
  finite-sample corrections or `"none"` for unadjusted clustered covariance

The estimator always requires complete treated and control support in the requested
event-time window for an admissible cohort.

### Stata parity validation

The default test suite does not execute Stata. On WSL machines with Windows Stata
available, run the opt-in parity check with:

```console
pytest -m stata
```

Set `STATA_EXE` if Stata is not installed at
`/mnt/c/Program Files/StataNow19/StataMP-64.exe`. The check requires the Windows Stata
environment to have `reghdfe`, `ftools`, and `require` installed.

## Quick start

```python
import pandas as pd

from stacked_eventstudy import (
    estimate_stacked_eventstudy,
    validate_stacked_eventstudy,
)

data = pd.DataFrame(
    {
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "age": [24, 25, 26, 24, 25, 26, 24, 25, 26],
        "treatment_age": [25, 25, 25, 26, 26, 26, 27, 27, 27],
        "outcome": [10.0, 10.5, 11.2, 10.1, 10.6, 10.9, 10.2, 10.7, 11.0],
        "calendar_year": [2004, 2005, 2006, 2004, 2005, 2006, 2004, 2005, 2006],
    }
)

validation = validate_stacked_eventstudy(
    data=data,
    id_col="id",
    age_col="age",
    treatment_age_col="treatment_age",
    outcome_col="outcome",
    l_min=-1,
    l_max=0,
    control_window=2,
    reference_event_time=-1,
    calendar_year_col="calendar_year",
)

print("Validation status:", validation.is_valid)
print("Validation errors:", validation.errors)

if validation.is_valid:
    result = estimate_stacked_eventstudy(
        data=data,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-1,
        l_max=0,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
        return_stacked_data=True,
    )

    print(result.cohort_params)
    print(result.average_params)
```

A larger executable example is available in
[examples/basic_usage.py](/home/zimpelmann/ECON/stacked_eventstudy/examples/basic_usage.py:1).

## Returned objects

`estimate_stacked_eventstudy(...)` returns a `StackedEventStudyResult`.

### `result.cohort_params`

One row per `subevent x event_time`, including:

- `subevent`
- `event_time`
- `term_label`
- `estimate`
- `std_error`
- `ci_low`
- `ci_high`
- `scale`
- `n_treated_individuals`
- `n_control_individuals`
- `n_treated_obs`
- `n_control_obs`

When `heterogeneity_col` is supplied, this table also includes `heterogeneity_col` and
`heterogeneity_value`.

### `result.average_params`

One row per event time, including:

- `event_time`
- `estimate`
- `std_error`
- `ci_low`
- `ci_high`
- `n_cohorts`
- `scale`

When `heterogeneity_col` is supplied, this table is one row per group and event time and
also includes `heterogeneity_col`, `heterogeneity_value`, and `weight_scheme`.

### `result.cohort_weights`

One row per admissible treated cohort:

- `subevent`
- `n_observations`
- `weight_mass`
- `weight`

`n_observations` counts focal treated cohort rows in the requested event-time window.
`weight_mass` is the corresponding sum of input weights; in unweighted data this equals
`n_observations`. The normalized `weight` is the cohort's `weight_mass` divided by the
total `weight_mass` across retained cohorts.

When `heterogeneity_col` is supplied, weights are returned by group and cohort.

### `result.vcov_average`

Covariance matrix for the aggregated event-study coefficients, indexed by event time.
When `heterogeneity_col` is supplied, rows and columns are indexed by
`heterogeneity_value` and `event_time`.

### `result.contrast_params`

Pairwise group differences for each event time when `heterogeneity_col` is supplied. The
table is empty otherwise.

### `result.stacked_data`

Returned only when `return_stacked_data=True`. This is useful for debugging cohort
construction and control alignment.

## Example interpretation

- `cohort_params` tells you how the estimated effect evolves within each admissible
  treatment-age cohort
- `average_params` tells you the weighted average effect across admissible cohorts at
  each event time
- `cohort_weights` tells you how much each cohort contributes to that average

## Warnings and limitations

- Controls are aligned to the treated cohort's event time, not their own.
- The post-birth horizon is constrained by `control_window`.
- Supported regression backends are `statsmodels` and `pyfixest`.
- There is no conventional benchmark estimator in the current package version.
- The current implementation is aimed at clean panel inputs and synthetic validation
  first; broader empirical hardening is still ongoing.

## Example script

Run the included example with:

```bash
PYTHONPATH=src pixi run python examples/basic_usage.py
```

Run the heterogeneity example with:

```bash
PYTHONPATH=src pixi run python examples/heterogeneity_usage.py
```

## Reference

Melentyeva, V., & Riedel, L. (2025). Child penalty estimation and mothers' age at first
birth (No. 25-033). ZEW Discussion Papers.
