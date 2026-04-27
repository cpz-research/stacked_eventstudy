"""Public package interface for stacked_eventstudy."""

from stacked_eventstudy.api import estimate_stacked_eventstudy, validate_stacked_eventstudy
from stacked_eventstudy.types import (
    EstimatorConfig,
    StackedEventStudyResult,
    StackedEventStudyValidation,
)

__all__ = [
    "EstimatorConfig",
    "StackedEventStudyResult",
    "StackedEventStudyValidation",
    "estimate_stacked_eventstudy",
    "validate_stacked_eventstudy",
]
