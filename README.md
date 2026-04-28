# stacked_eventstudy

`stacked_eventstudy` implements a stacked difference-in-differences estimator with
rolling-window controls by age at first birth.

The current package is built around the heterogeneity-robust stacked estimator described
in Melentyeva and Riedel, where each treatment-age cohort is estimated in its own
stacked subevent and the resulting event-study coefficients are aggregated using treated
cohort shares.

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
1. aggregates cohort-specific coefficients using treated cohort shares

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
    balance=True,
    scale="none",
    backend="statsmodels",
    return_stacked_data=False,
)
```

Important arguments:

- `control_window`: width of the future-treated control window
- `reference_event_time`: omitted event time, usually `-1`
- `balance`: enforces complete treated and control support in the requested window
- `backend`: regression backend, either `"statsmodels"` or `"pyfixest"`
- `scale="pre_birth"`: rescales effects by the treated cohort's mean outcome at the
  reference period
- `cluster_col`: overrides default clustering on the original individual id

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

### `result.average_params`

One row per event time, including:

- `event_time`
- `estimate`
- `std_error`
- `ci_low`
- `ci_high`
- `n_cohorts`
- `scale`

### `result.cohort_weights`

One row per admissible treated cohort:

- `subevent`
- `n_individuals`
- `weight`

### `result.vcov_average`

Covariance matrix for the aggregated event-study coefficients, indexed by event time.

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
