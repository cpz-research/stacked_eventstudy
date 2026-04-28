"""Example with a generic heterogeneity column."""

import pandas as pd

from stacked_eventstudy import estimate_stacked_eventstudy, validate_stacked_eventstudy


def main() -> None:
    """Run a stacked event-study example with group-specific effects."""
    rows: list[dict[str, int | float | str]] = []
    group_effects = {"a": 1.0, "b": 3.0, "c": 5.0}
    treatment_ages = (25, 26, 27, 28)
    unit_id = 1
    for segment, treatment_effect in group_effects.items():
        for treatment_age in treatment_ages:
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
                            "segment": segment,
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
        heterogeneity_col="segment",
    )

    print("Validation status:", validation.is_valid)
    print("Validation errors:", validation.errors)

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
        heterogeneity_col="segment",
        heterogeneity_weighting="within",
        backend="pyfixest",
        return_stacked_data=True,
    )

    print("\nGroup-specific average effects")
    print(
        result.average_params.pivot(
            index="event_time",
            columns="heterogeneity_value",
            values="estimate",
        ),
    )

    print("\nPairwise group contrasts")
    print(result.contrast_params)

    print("\nGroup-specific cohort weights")
    print(result.cohort_weights)


if __name__ == "__main__":
    main()
