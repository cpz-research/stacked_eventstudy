"""Basic usage example for the stacked_eventstudy package."""

import pandas as pd

from stacked_eventstudy import estimate_stacked_eventstudy, validate_stacked_eventstudy


def main() -> None:
    """Run a small stacked event-study example."""
    rows: list[dict[str, int | float]] = []
    treatment_effects = {25: 1.0, 26: 2.0, 27: 3.0, 28: 4.0}
    unit_id = 1
    for treatment_age, treatment_effect in treatment_effects.items():
        for _replicate in range(2):
            for age in range(22, 30):
                outcome = 5.0 + 0.5 * age
                if age >= treatment_age:
                    outcome += treatment_effect
                rows.append(
                    {
                        "id": unit_id,
                        "age": age,
                        "treatment_age": treatment_age,
                        "calendar_year": 2000 + age,
                        "outcome": outcome,
                    },
                )
            unit_id += 1

    data = pd.DataFrame(rows)

    validation = validate_stacked_eventstudy(
        data=data,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
    )

    print("Validation status:", validation.is_valid)
    print("Validation errors:", validation.errors)
    print(validation.cohort_diagnostics)

    result = estimate_stacked_eventstudy(
        data=data,
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-2,
        l_max=1,
        control_window=2,
        reference_event_time=-1,
        calendar_year_col="calendar_year",
        return_stacked_data=True,
    )

    print("\nCohort-specific effects")
    print(result.cohort_params)

    print("\nAverage effects")
    print(result.average_params)

    print("\nCohort weights")
    print(result.cohort_weights)

    print("\nAggregated covariance")
    print(result.vcov_average)


if __name__ == "__main__":
    main()
