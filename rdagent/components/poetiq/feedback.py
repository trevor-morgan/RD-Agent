"""Soft scoring system for Poetiq strategies.

Provides continuous scores (0.0-1.0) instead of binary decisions,
enabling more nuanced experiment evaluation and selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdagent.core.proposal import HypothesisFeedback


@dataclass
class SoftScore:
    """Continuous score in [0.0, 1.0] with metadata.

    Attributes:
        value: Primary score in [0.0, 1.0]
        confidence: Confidence in this score (default 1.0)
        components: Sub-scores by metric name for detailed analysis
    """

    value: float
    confidence: float = 1.0
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Clamp value to [0.0, 1.0]."""
        self.value = max(0.0, min(1.0, self.value))
        self.confidence = max(0.0, min(1.0, self.confidence))

    @property
    def decision(self) -> bool:
        """Binary decision for backward compatibility."""
        from rdagent.components.poetiq.conf import POETIQ_SETTINGS

        return self.value >= POETIQ_SETTINGS.score_threshold

    def __str__(self) -> str:
        components_str = ", ".join(f"{k}={v:.4f}" for k, v in self.components.items())
        return f"SoftScore(value={self.value:.4f}, confidence={self.confidence:.2f}, components=[{components_str}])"


class ScoredHypothesisFeedback(HypothesisFeedback):
    """Extended hypothesis feedback with soft scoring support.

    Maintains backward compatibility with binary `decision` while
    supporting continuous soft scores for more nuanced evaluation.
    """

    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        *,
        soft_score: SoftScore | None = None,
        code_change_summary: str | None = None,
        decision: bool | None = None,
        eda_improvement: str | None = None,
        acceptable: bool | None = None,
    ) -> None:
        # If soft_score provided, derive decision from it
        if soft_score is not None and decision is None:
            decision = soft_score.decision
        elif decision is None:
            decision = False

        super().__init__(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            code_change_summary=code_change_summary,
            decision=decision,
            eda_improvement=eda_improvement,
            acceptable=acceptable,
        )

        # Store soft score, creating default if not provided
        self.soft_score = soft_score or SoftScore(value=1.0 if decision else 0.0)

    @classmethod
    def from_feedback(
        cls,
        feedback: HypothesisFeedback,
        score: float | None = None,
        soft_score: SoftScore | None = None,
    ) -> ScoredHypothesisFeedback:
        """Convert existing HypothesisFeedback to scored version.

        Args:
            feedback: Original feedback to convert
            score: Optional score value (creates SoftScore with this value)
            soft_score: Optional complete SoftScore object (takes precedence)

        Returns:
            ScoredHypothesisFeedback with soft scoring
        """
        if soft_score is None:
            if score is not None:
                soft_score = SoftScore(value=score)
            else:
                soft_score = SoftScore(value=1.0 if feedback.decision else 0.0)

        return cls(
            observations=feedback.observations,
            hypothesis_evaluation=feedback.hypothesis_evaluation,
            new_hypothesis=feedback.new_hypothesis,
            reason=feedback.reason,
            code_change_summary=feedback.code_change_summary,
            decision=feedback.decision,
            eda_improvement=feedback.eda_improvement,
            acceptable=feedback.acceptable,
            soft_score=soft_score,
        )

    def __str__(self) -> str:
        return f"""{super().__str__()}
Soft Score: {self.soft_score}"""


def compute_soft_score(
    result: dict[str, Any] | Any,
    baseline: dict[str, Any] | Any | None = None,
    metric: str | None = None,
) -> SoftScore:
    """Compute soft score from experiment results.

    Args:
        result: Experiment result dict or DataFrame-like object
        baseline: Optional baseline result for comparison
        metric: Primary metric to use (defaults to POETIQ_SETTINGS.score_metric)

    Returns:
        SoftScore with normalized value and component scores
    """
    from rdagent.components.poetiq.conf import POETIQ_SETTINGS

    if metric is None:
        metric = POETIQ_SETTINGS.score_metric

    try:
        value = _extract_metric_value(result, metric)
        if value is None:
            return SoftScore(value=0.0, confidence=0.3, components={metric: "missing"})

        # Normalize based on metric type
        normalized = _normalize_metric(value, metric)

        # Handle baseline comparison if provided
        if baseline is not None:
            try:
                if hasattr(baseline, "get"):
                    baseline_value = float(baseline.get(metric, 0))
                elif hasattr(baseline, "loc"):
                    baseline_value = float(
                        baseline.loc[metric].iloc[0] if hasattr(baseline.loc[metric], "iloc") else baseline.loc[metric]
                    )
                else:
                    baseline_value = float(baseline)

                # Compute improvement ratio
                if baseline_value != 0:
                    improvement = (value - baseline_value) / abs(baseline_value)
                    # Boost score if improving, penalize if degrading
                    normalized = min(1.0, max(0.0, normalized + improvement * 0.2))
            except (KeyError, ValueError, TypeError):
                pass

        return SoftScore(
            value=normalized,
            confidence=1.0,
            components={metric: value},
        )

    except (KeyError, ValueError, TypeError, AttributeError) as e:
        # Return zero score on any error
        return SoftScore(value=0.0, confidence=0.5, components={"error": str(e)})


def _extract_metric_value(result: Any, metric: str) -> float | None:
    """Extract a metric value from various result container types."""
    if result is None:
        return None

    # Mapping-like
    if hasattr(result, "get"):
        try:
            val = result.get(metric)
            if val is not None:
                return float(val)
        except Exception:
            pass

    # DataFrame-like
    if hasattr(result, "loc"):
        try:
            loc_val = result.loc[metric]
            if hasattr(loc_val, "iloc"):
                loc_val = loc_val.iloc[0]
            return float(loc_val)
        except Exception:
            pass

    # Sequence of pairs / nested mapping
    if hasattr(result, "items"):
        try:
            for key, val in result.items():
                if metric in str(key):
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if metric in str(sub_key):
                                return float(sub_val)
                    else:
                        return float(val)
        except Exception:
            pass

    # Simple numeric
    try:
        return float(result)
    except Exception:
        return None


def _normalize_metric(value: float, metric: str) -> float:
    """Normalize metric value to [0.0, 1.0] based on metric type.

    Args:
        value: Raw metric value
        metric: Metric name

    Returns:
        Normalized value in [0.0, 1.0]
    """
    metric_lower = metric.lower()

    # Information Coefficient: typically in [-0.1, 0.1], higher is better
    if metric_lower == "ic":
        return min(1.0, max(0.0, (value + 0.1) / 0.2))

    # IC Information Ratio: typically in [-1, 3], higher is better
    if metric_lower == "icir":
        return min(1.0, max(0.0, (value + 1) / 4))

    # Rank IC: similar to IC
    if metric_lower in ("rank_ic", "rankic"):
        return min(1.0, max(0.0, (value + 0.1) / 0.2))

    # Sharpe ratio: typically in [-1, 3], higher is better
    if metric_lower == "sharpe":
        return min(1.0, max(0.0, (value + 1) / 4))

    # Annualized return: typically in [-0.5, 0.5], higher is better
    if "annualized_return" in metric_lower or "arr" in metric_lower:
        return min(1.0, max(0.0, (value + 0.5) / 1.0))

    # Max drawdown: typically in [-1, 0], closer to 0 is better (less negative)
    if "max_drawdown" in metric_lower or "mdd" in metric_lower:
        return min(1.0, max(0.0, (value + 1) / 1.0))

    # Default: assume value is already in reasonable range
    # If negative, shift to positive
    if value < 0:
        return min(1.0, max(0.0, value + 1))

    return min(1.0, max(0.0, value))
