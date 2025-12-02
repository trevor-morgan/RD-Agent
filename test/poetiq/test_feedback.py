"""Tests for Poetiq soft scoring system."""

import pytest

from rdagent.components.poetiq.feedback import (
    ScoredHypothesisFeedback,
    SoftScore,
    compute_soft_score,
    _normalize_metric,
)
from rdagent.core.proposal import HypothesisFeedback


class TestSoftScore:
    """Tests for SoftScore dataclass."""

    def test_value_clamping(self):
        """Values should be clamped to [0.0, 1.0]."""
        score = SoftScore(value=1.5)
        assert score.value == 1.0

        score = SoftScore(value=-0.5)
        assert score.value == 0.0

        score = SoftScore(value=0.5)
        assert score.value == 0.5

    def test_confidence_clamping(self):
        """Confidence should be clamped to [0.0, 1.0]."""
        score = SoftScore(value=0.5, confidence=1.5)
        assert score.confidence == 1.0

        score = SoftScore(value=0.5, confidence=-0.5)
        assert score.confidence == 0.0

    def test_decision_property(self):
        """Decision should be True when value >= threshold."""
        # Default threshold is 0.5
        score = SoftScore(value=0.6)
        assert score.decision is True

        score = SoftScore(value=0.4)
        assert score.decision is False

        score = SoftScore(value=0.5)
        assert score.decision is True

    def test_components(self):
        """Components dict should store metric values."""
        score = SoftScore(value=0.7, components={"IC": 0.05, "Sharpe": 1.2})
        assert score.components["IC"] == 0.05
        assert score.components["Sharpe"] == 1.2

    def test_str_representation(self):
        """String representation should include all fields."""
        score = SoftScore(value=0.7, confidence=0.9, components={"IC": 0.05})
        s = str(score)
        assert "0.7" in s
        assert "0.9" in s or "0.90" in s
        assert "IC" in s


class TestScoredHypothesisFeedback:
    """Tests for ScoredHypothesisFeedback class."""

    def test_creation_with_soft_score(self):
        """Should create feedback with explicit soft score."""
        soft_score = SoftScore(value=0.8)
        feedback = ScoredHypothesisFeedback(
            observations="Test observations",
            hypothesis_evaluation="Test evaluation",
            new_hypothesis="Test new hypothesis",
            reason="Test reason",
            soft_score=soft_score,
        )
        assert feedback.soft_score.value == 0.8
        assert feedback.decision is True

    def test_creation_without_soft_score(self):
        """Should create default soft score from decision."""
        feedback = ScoredHypothesisFeedback(
            observations="Test",
            hypothesis_evaluation="Test",
            new_hypothesis="Test",
            reason="Test",
            decision=True,
        )
        assert feedback.soft_score.value == 1.0

        feedback = ScoredHypothesisFeedback(
            observations="Test",
            hypothesis_evaluation="Test",
            new_hypothesis="Test",
            reason="Test",
            decision=False,
        )
        assert feedback.soft_score.value == 0.0

    def test_from_feedback_conversion(self):
        """Should convert HypothesisFeedback to scored version."""
        original = HypothesisFeedback(
            observations="Test",
            hypothesis_evaluation="Test",
            new_hypothesis="Test",
            reason="Test",
            decision=True,
        )

        scored = ScoredHypothesisFeedback.from_feedback(original, score=0.75)
        assert scored.soft_score.value == 0.75
        assert scored.observations == "Test"
        assert scored.decision is True

    def test_from_feedback_with_soft_score(self):
        """Should accept complete SoftScore object."""
        original = HypothesisFeedback(
            observations="Test",
            hypothesis_evaluation="Test",
            new_hypothesis="Test",
            reason="Test",
            decision=True,
        )

        soft_score = SoftScore(value=0.9, components={"IC": 0.08})
        scored = ScoredHypothesisFeedback.from_feedback(original, soft_score=soft_score)
        assert scored.soft_score.value == 0.9
        assert scored.soft_score.components["IC"] == 0.08


class TestNormalizeMetric:
    """Tests for metric normalization."""

    def test_ic_normalization(self):
        """IC should be normalized from [-0.1, 0.1] to [0, 1]."""
        # IC = 0.1 -> normalized = 1.0
        assert _normalize_metric(0.1, "IC") == pytest.approx(1.0)
        # IC = -0.1 -> normalized = 0.0
        assert _normalize_metric(-0.1, "IC") == pytest.approx(0.0)
        # IC = 0 -> normalized = 0.5
        assert _normalize_metric(0.0, "IC") == pytest.approx(0.5)
        # IC = 0.05 -> normalized = 0.75
        assert _normalize_metric(0.05, "IC") == pytest.approx(0.75)

    def test_sharpe_normalization(self):
        """Sharpe should be normalized from [-1, 3] to [0, 1]."""
        # Sharpe = 3 -> normalized = 1.0
        assert _normalize_metric(3.0, "Sharpe") == pytest.approx(1.0)
        # Sharpe = -1 -> normalized = 0.0
        assert _normalize_metric(-1.0, "Sharpe") == pytest.approx(0.0)
        # Sharpe = 1 -> normalized = 0.5
        assert _normalize_metric(1.0, "Sharpe") == pytest.approx(0.5)

    def test_unknown_metric_default(self):
        """Unknown metrics should use default normalization."""
        # Positive value in [0, 1] range
        assert _normalize_metric(0.5, "unknown") == pytest.approx(0.5)
        # Negative value gets shifted
        assert _normalize_metric(-0.3, "unknown") == pytest.approx(0.7)


class TestComputeSoftScore:
    """Tests for compute_soft_score function."""

    def test_dict_result(self):
        """Should extract metric from dict result."""
        result = {"IC": 0.05, "Sharpe": 1.0}
        score = compute_soft_score(result, metric="IC")
        assert score.value == pytest.approx(0.75)
        assert score.components["IC"] == 0.05

    def test_missing_metric(self):
        """Should return zero score for missing metric."""
        result = {"Sharpe": 1.0}
        score = compute_soft_score(result, metric="IC")
        # Missing metric defaults to 0, which normalizes to 0.5
        assert score.value == pytest.approx(0.5)

    def test_invalid_result(self):
        """Should return zero score with low confidence on error."""
        score = compute_soft_score(None, metric="IC")
        assert score.value == 0.0
        assert score.confidence == 0.5
