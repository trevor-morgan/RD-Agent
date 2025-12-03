"""Ensemble model utilities for combining multiple predictors.

This module provides reusable components for building adaptive ensembles:

- **WeightingMethod**: Enum for weighting strategies (equal, IC, Sharpe, learned)
- **PerformanceTracker**: EMA-based performance tracking for online adaptation
- **DiversityRegularizer**: Encourages diverse predictions across models

These utilities are framework-agnostic and can be used with any model type.

Example:
    >>> from rdagent_lab.models.ensemble import PerformanceTracker, WeightingMethod
    >>> tracker = PerformanceTracker(n_models=3)
    >>> weights = tracker.get_weights(WeightingMethod.IC_WEIGHTED)
"""

from rdagent_lab.models.ensemble.adaptive_ensemble import (
    WeightingMethod,
    PerformanceTracker,
    DiversityRegularizer,
)

__all__ = [
    "WeightingMethod",
    "PerformanceTracker",
    "DiversityRegularizer",
]
