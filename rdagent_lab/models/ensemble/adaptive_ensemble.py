"""Adaptive ensemble utilities for combining multiple models.

Provides reusable components for building adaptive ensembles with
dynamically adjusted weights based on recent performance.

Components:
    - WeightingMethod: Enum for weighting strategies
    - PerformanceTracker: EMA-based performance tracking
    - DiversityRegularizer: Encourages diverse predictions

Example:
    >>> import pandas as pd
    >>> from rdagent_lab.models.ensemble import PerformanceTracker, WeightingMethod
    >>>
    >>> # Track 3 models
    >>> tracker = PerformanceTracker(n_models=3, lookback=60, ema_alpha=0.1)
    >>>
    >>> # Update with predictions and labels
    >>> preds = pd.Series([0.1, 0.2, 0.15])
    >>> labels = pd.Series([0.12, 0.18, 0.14])
    >>> tracker.update(model_idx=0, predictions=preds, labels=labels)
    >>>
    >>> # Get adaptive weights
    >>> weights = tracker.get_weights(WeightingMethod.IC_WEIGHTED)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class WeightingMethod(str, Enum):
    """Available weighting methods for ensemble combination.

    Attributes:
        EQUAL: Equal weights for all models
        IC_WEIGHTED: Weight by Information Coefficient (correlation with returns)
        SHARPE_WEIGHTED: Weight by rolling Sharpe ratio
        LEARNED: Placeholder for learned/meta-model weights
    """

    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    SHARPE_WEIGHTED = "sharpe_weighted"
    LEARNED = "learned"


@dataclass
class PerformanceTracker:
    """Tracks model performance over time for adaptive weighting.

    Maintains rolling statistics for each base model using exponential
    moving averages, enabling weight adaptation without full retraining.

    Attributes:
        n_models: Number of models to track
        lookback: Window size for rolling statistics
        ema_alpha: Decay rate for exponential moving average (0-1)
        ic_history: Per-model IC history
        returns_history: Per-model returns history
        ema_ic: Current EMA of IC for each model
        ema_sharpe: Current EMA of Sharpe for each model

    Example:
        >>> tracker = PerformanceTracker(n_models=3)
        >>> tracker.update(0, predictions, labels)
        >>> weights = tracker.get_weights(WeightingMethod.IC_WEIGHTED)
    """

    n_models: int
    lookback: int = 60
    ema_alpha: float = 0.1

    # Per-model tracking (initialized in __post_init__)
    ic_history: list[list[float]] = field(default_factory=list)
    returns_history: list[list[float]] = field(default_factory=list)

    # Current EMA values
    ema_ic: np.ndarray = field(init=False)
    ema_sharpe: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initialize tracking arrays."""
        self.ic_history = [[] for _ in range(self.n_models)]
        self.returns_history = [[] for _ in range(self.n_models)]
        self.ema_ic = np.ones(self.n_models) / self.n_models
        self.ema_sharpe = np.ones(self.n_models) / self.n_models

    def update(
        self,
        model_idx: int,
        predictions: pd.Series,
        labels: pd.Series,
    ) -> None:
        """Update performance metrics for a model.

        Args:
            model_idx: Index of the model (0 to n_models-1)
            predictions: Model predictions for the period
            labels: Actual labels/returns for the period
        """
        # Compute IC (Information Coefficient)
        ic = predictions.corr(labels)
        if np.isnan(ic):
            ic = 0.0

        self.ic_history[model_idx].append(ic)
        if len(self.ic_history[model_idx]) > self.lookback:
            self.ic_history[model_idx] = self.ic_history[model_idx][-self.lookback:]

        # Update EMA
        self.ema_ic[model_idx] = (
            self.ema_alpha * ic + (1 - self.ema_alpha) * self.ema_ic[model_idx]
        )

        # Returns tracking (for Sharpe calculation)
        ret = predictions.mean()
        self.returns_history[model_idx].append(ret)
        if len(self.returns_history[model_idx]) > self.lookback:
            self.returns_history[model_idx] = self.returns_history[model_idx][-self.lookback:]

        # Update Sharpe EMA
        if len(self.returns_history[model_idx]) >= 2:
            rets = np.array(self.returns_history[model_idx])
            sharpe = rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)
            sharpe = np.clip(sharpe, -5, 5)  # Bound extreme values
        else:
            sharpe = 0.0

        self.ema_sharpe[model_idx] = (
            self.ema_alpha * sharpe + (1 - self.ema_alpha) * self.ema_sharpe[model_idx]
        )

    def get_weights(self, method: WeightingMethod) -> np.ndarray:
        """Get current model weights based on method.

        Args:
            method: Weighting strategy to use

        Returns:
            Normalized weight vector of shape (n_models,)
        """
        if method == WeightingMethod.EQUAL:
            weights = np.ones(self.n_models)

        elif method == WeightingMethod.IC_WEIGHTED:
            # Weight by positive IC, zero out negative
            weights = np.maximum(self.ema_ic, 0)
            if weights.sum() < 1e-8:
                weights = np.ones(self.n_models)

        elif method == WeightingMethod.SHARPE_WEIGHTED:
            # Weight by positive Sharpe, zero out negative
            weights = np.maximum(self.ema_sharpe, 0)
            if weights.sum() < 1e-8:
                weights = np.ones(self.n_models)

        else:
            weights = np.ones(self.n_models)

        # Normalize
        return weights / weights.sum()

    def reset(self) -> None:
        """Reset all tracking state."""
        self.__post_init__()


class DiversityRegularizer:
    """Encourages diverse predictions across models.

    Penalizes ensembles where models produce identical predictions,
    which would defeat the purpose of ensembling.

    Attributes:
        penalty_weight: Weight for diversity penalty (0-1)

    Example:
        >>> regularizer = DiversityRegularizer(penalty_weight=0.1)
        >>> penalty = regularizer.compute_penalty(predictions_list)
        >>> bonus = regularizer.compute_diversity_bonus(predictions_list)
    """

    def __init__(self, penalty_weight: float = 0.1):
        """Initialize regularizer.

        Args:
            penalty_weight: Weight for diversity penalty (higher = more penalty)
        """
        self.penalty_weight = penalty_weight

    def compute_penalty(self, predictions: list[pd.Series]) -> float:
        """Compute diversity penalty.

        Higher penalty = more correlated predictions (bad for diversity).

        Args:
            predictions: List of prediction series from each model

        Returns:
            Penalty value (0 = perfectly diverse, 1 = identical)
        """
        n = len(predictions)
        if n < 2:
            return 0.0

        # Compute pairwise correlations
        total_corr = 0.0
        n_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                corr = predictions[i].corr(predictions[j])
                if not np.isnan(corr):
                    total_corr += abs(corr)
                    n_pairs += 1

        if n_pairs == 0:
            return 0.0

        avg_corr = total_corr / n_pairs
        return self.penalty_weight * avg_corr

    def compute_diversity_bonus(self, predictions: list[pd.Series]) -> np.ndarray:
        """Compute per-model diversity bonus for weight adjustment.

        Models that are more diverse (less correlated with others) get higher bonus.

        Args:
            predictions: List of prediction series from each model

        Returns:
            Diversity bonus per model, shape (n_models,)
        """
        n = len(predictions)
        if n < 2:
            return np.ones(n)

        # For each model, compute average correlation with others
        avg_corr_per_model = np.zeros(n)

        for i in range(n):
            corrs = []
            for j in range(n):
                if i != j:
                    corr = predictions[i].corr(predictions[j])
                    if not np.isnan(corr):
                        corrs.append(abs(corr))
            if corrs:
                avg_corr_per_model[i] = np.mean(corrs)

        # Bonus = inverse of correlation (more diverse = higher bonus)
        bonus = 1 - avg_corr_per_model
        return bonus


__all__ = [
    "WeightingMethod",
    "PerformanceTracker",
    "DiversityRegularizer",
]
