# stacked_eventstudy

`stacked_eventstudy` implements the stacked difference-in-differences estimator with
rolling-window controls by age at first birth outlined in Melentyeva, Valentina, and
Lukas Riedel. "Child Penalty Estimation and Mothers' Age at First Birth." (2025)

The package is designed for individual-level panel data where all individuals are
eventually treated and age is observed in integer years.

## Status

The initial package scaffold includes:

- public APIs for validation and estimation
- structured result dataclasses
- data validation
- stack construction
- a stacked event-study estimator using a formula-based regression backend

## Main Functions

- `validate_stacked_eventstudy(...)`
- `estimate_stacked_eventstudy(...)`

## Development

Use `pixi` for all environment management and test execution.
