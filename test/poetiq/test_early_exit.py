"""Tests for Poetiq early exit checker."""

from unittest.mock import Mock, patch

import pytest

from rdagent.components.poetiq.early_exit import EarlyExitChecker


class MockExperiment:
    """Mock experiment for testing."""

    def __init__(self, result: dict = None):
        self.result = result


class MockFeedback:
    """Mock feedback for testing."""

    def __init__(self, decision: bool = True):
        self.decision = decision


class MockTrace:
    """Mock trace for testing."""

    def __init__(self, hist: list = None):
        self.hist = hist or []


class TestEarlyExitChecker:
    """Tests for EarlyExitChecker."""

    def test_no_exit_when_not_configured(self):
        """Should not exit when early_exit_metric is None."""
        checker = EarlyExitChecker()

        exp = MockExperiment(result={"IC": 0.1})
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = None
            mock_settings.early_exit_threshold = None

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is False
        assert reason == ""

    def test_exit_when_threshold_met_higher(self):
        """Should exit when metric exceeds threshold (higher is better)."""
        checker = EarlyExitChecker()

        exp = MockExperiment(result={"IC": 0.08})
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is True
        assert "IC" in reason
        assert "0.08" in reason

    def test_no_exit_when_below_threshold(self):
        """Should not exit when metric is below threshold."""
        checker = EarlyExitChecker()

        exp = MockExperiment(result={"IC": 0.03})
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is False

    def test_exit_when_threshold_met_lower(self):
        """Should exit when metric is below threshold (lower is better).

        For direction='lower': exit when value <= threshold.
        Use case: exit when validation loss is low enough.
        """
        checker = EarlyExitChecker()

        # Validation loss - lower is better
        # value=0.01 is below threshold=0.05, so should exit
        exp = MockExperiment(result={"val_loss": 0.01})
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "val_loss"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "lower"

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is True
        assert "val_loss" in reason

    def test_no_exit_on_failed_experiment(self):
        """Should not exit when experiment failed (decision=False)."""
        checker = EarlyExitChecker()

        exp = MockExperiment(result={"IC": 0.1})
        fb = MockFeedback(decision=False)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is False

    def test_no_exit_on_empty_trace(self):
        """Should not exit when trace is empty."""
        checker = EarlyExitChecker()
        trace = MockTrace(hist=[])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is False

    def test_no_exit_when_result_is_none(self):
        """Should not exit when experiment has no result."""
        checker = EarlyExitChecker()

        exp = MockExperiment(result=None)
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            should_exit, reason = checker.should_exit(trace)

        assert should_exit is False

    def test_consecutive_success_tracking(self):
        """Should track consecutive successes."""
        checker = EarlyExitChecker()

        exp = MockExperiment(result={"IC": 0.08})
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            checker.should_exit(trace)
            assert checker._consecutive_successes == 1

            checker.should_exit(trace)
            assert checker._consecutive_successes == 2

    def test_reset_consecutive_on_failure(self):
        """Should reset consecutive counter on failure."""
        checker = EarlyExitChecker()
        checker._consecutive_successes = 3

        exp = MockExperiment(result={"IC": 0.02})  # Below threshold
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            checker.should_exit(trace)

        assert checker._consecutive_successes == 0

    def test_get_status(self):
        """Should return current status."""
        checker = EarlyExitChecker()
        checker._consecutive_successes = 2

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.early_exit_metric = "IC"
            mock_settings.early_exit_threshold = 0.05
            mock_settings.early_exit_direction = "higher"

            status = checker.get_status()

        assert status["metric"] == "IC"
        assert status["threshold"] == 0.05
        assert status["direction"] == "higher"
        assert status["consecutive_successes"] == 2

    def test_reset(self):
        """Should reset consecutive counter."""
        checker = EarlyExitChecker()
        checker._consecutive_successes = 5

        checker.reset()

        assert checker._consecutive_successes == 0
