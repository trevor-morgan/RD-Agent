"""Early exit checker for Poetiq.

Stops exploration when experiments meet success threshold,
saving compute on already-solved problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.core.proposal import Trace


class EarlyExitChecker:
    """Check if experiment results meet early exit threshold.

    Monitors experiment metrics and signals when a threshold is exceeded,
    allowing the loop to terminate early on success.
    """

    def __init__(self) -> None:
        """Initialize early exit checker."""
        self._consecutive_successes: int = 0

    def should_exit(self, trace: Trace) -> tuple[bool, str]:
        """Check if early exit conditions are met.

        Args:
            trace: Experiment trace history

        Returns:
            Tuple of (should_exit, reason_string)
        """
        from rdagent.components.poetiq.conf import POETIQ_SETTINGS

        metric = POETIQ_SETTINGS.early_exit_metric
        threshold = POETIQ_SETTINGS.early_exit_threshold

        # Check if early exit is configured
        if not metric or threshold is None:
            return False, ""

        if not trace.hist:
            return False, ""

        # Get latest experiment
        exp, fb = trace.hist[-1]

        # Only check successful experiments
        if not fb.decision:
            self._consecutive_successes = 0
            return False, ""

        if exp.result is None:
            return False, ""

        # Extract metric value
        try:
            value = self._extract_metric(exp.result, metric)
            if value is None:
                return False, ""

            direction = POETIQ_SETTINGS.early_exit_direction

            # Check threshold
            threshold_met = False
            if (direction == "higher" and value >= threshold) or (direction == "lower" and value <= threshold):
                threshold_met = True

            if threshold_met:
                self._consecutive_successes += 1
                reason = (
                    f"Early exit: {metric}={value:.4f} "
                    f"{'≥' if direction == 'higher' else '≤'} {threshold} "
                    f"(consecutive: {self._consecutive_successes})"
                )
                return True, reason
            self._consecutive_successes = 0
            return False, ""

        except (KeyError, ValueError, TypeError, AttributeError):
            return False, ""

    def _extract_metric(self, result: object, metric: str) -> float | None:
        """Extract metric value from experiment result.

        Args:
            result: Experiment result (dict or DataFrame-like)
            metric: Metric name to extract

        Returns:
            Metric value or None if not found
        """
        # Handle dict-like results
        if hasattr(result, "get"):
            value = result.get(metric)
            if value is not None:
                return float(value)

        # Handle DataFrame-like results
        if hasattr(result, "loc"):
            try:
                loc_result = result.loc[metric]
                if hasattr(loc_result, "iloc"):
                    return float(loc_result.iloc[0])
                return float(loc_result)
            except (KeyError, IndexError):
                pass

        # Handle nested dicts (e.g., result["1day"]["IC"])
        if hasattr(result, "items"):
            for key, val in result.items():
                if metric in str(key):
                    if isinstance(val, dict):
                        for k, v in val.items():
                            if metric in str(k):
                                return float(v)
                    else:
                        return float(val)

        return None

    def reset(self) -> None:
        """Reset consecutive success counter."""
        self._consecutive_successes = 0

    def get_status(self) -> dict[str, int | str | float | None]:
        """Get current early exit status.

        Returns:
            Dict with current configuration and state
        """
        from rdagent.components.poetiq.conf import POETIQ_SETTINGS

        return {
            "metric": POETIQ_SETTINGS.early_exit_metric,
            "threshold": POETIQ_SETTINGS.early_exit_threshold,
            "direction": POETIQ_SETTINGS.early_exit_direction,
            "consecutive_successes": self._consecutive_successes,
        }
