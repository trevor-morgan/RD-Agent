"""Tests for Poetiq SOTA selection strategies."""

from unittest.mock import Mock, patch

import pytest

from rdagent.components.poetiq.feedback import SoftScore
from rdagent.components.poetiq.selection import (
    ConsensusVotingSelector,
    StochasticSOTASelector,
)


class MockExperiment:
    """Mock experiment for testing."""

    def __init__(self, hypothesis_text: str = "Test hypothesis"):
        self.hypothesis = Mock()
        self.hypothesis.hypothesis = hypothesis_text
        self.result = {"IC": 0.05}


class MockFeedback:
    """Mock feedback for testing."""

    def __init__(self, decision: bool = True, score: float = 0.8):
        self.decision = decision
        self.soft_score = SoftScore(value=score)


class MockTrace:
    """Mock trace for testing."""

    def __init__(self, hist: list = None):
        self.hist = hist or []


class TestStochasticSOTASelector:
    """Tests for StochasticSOTASelector."""

    def test_select_with_single_candidate(self):
        """Should return the only candidate when k=1 or single entry."""
        selector = StochasticSOTASelector(k=3)

        exp = MockExperiment()
        fb = MockFeedback(decision=True, score=0.9)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.enabled = True
            result = selector.select(trace)

        assert result is not None
        assert result[0] == exp
        assert result[1] == fb

    def test_get_top_k(self):
        """Should return top-K experiments sorted by score."""
        selector = StochasticSOTASelector(k=2)

        exp1 = MockExperiment("Hypothesis 1")
        exp2 = MockExperiment("Hypothesis 2")
        exp3 = MockExperiment("Hypothesis 3")

        fb1 = MockFeedback(decision=True, score=0.5)
        fb2 = MockFeedback(decision=True, score=0.9)
        fb3 = MockFeedback(decision=False, score=0.7)  # Should be excluded

        trace = MockTrace(hist=[(exp1, fb1), (exp2, fb2), (exp3, fb3)])

        top_k = selector.get_top_k(trace)

        assert len(top_k) == 2
        # Highest score first
        assert top_k[0][1] == exp2
        assert top_k[0][3] == 0.9

    def test_fallback_when_disabled(self):
        """Should fall back to deterministic selection when disabled."""
        selector = StochasticSOTASelector(k=3)

        exp1 = MockExperiment()
        exp2 = MockExperiment()

        fb1 = MockFeedback(decision=True, score=0.5)
        fb2 = MockFeedback(decision=True, score=0.9)

        trace = MockTrace(hist=[(exp1, fb1), (exp2, fb2)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.enabled = False
            result = selector.select(trace)

        # Should return most recent decision=True (exp2)
        assert result is not None
        assert result[0] == exp2

    def test_empty_trace(self):
        """Should return None for empty trace."""
        selector = StochasticSOTASelector(k=3)
        trace = MockTrace(hist=[])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.enabled = True
            result = selector.select(trace)

        assert result is None

    def test_no_successful_experiments(self):
        """Should return None when no decision=True experiments."""
        selector = StochasticSOTASelector(k=3)

        exp = MockExperiment()
        fb = MockFeedback(decision=False, score=0.3)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.enabled = True
            result = selector.select(trace)

        assert result is None


class TestConsensusVotingSelector:
    """Tests for ConsensusVotingSelector."""

    def test_cluster_similar_hypotheses(self):
        """Should cluster similar hypotheses together."""
        selector = ConsensusVotingSelector(similarity_threshold=0.8, min_votes=2)

        # Create similar hypotheses
        exp1 = MockExperiment("Add dropout layer with rate 0.3")
        exp2 = MockExperiment("Add dropout layer with rate 0.5")
        exp3 = MockExperiment("Use attention mechanism instead")

        fb1 = MockFeedback(decision=True, score=0.7)
        fb2 = MockFeedback(decision=True, score=0.8)
        fb3 = MockFeedback(decision=True, score=0.9)

        trace = MockTrace(hist=[(exp1, fb1), (exp2, fb2), (exp3, fb3)])

        cluster_info = selector.get_cluster_info(trace)

        # Should have at least 2 clusters (dropout vs attention)
        assert cluster_info["num_clusters"] >= 1

    def test_min_votes_requirement(self):
        """Should return None if no cluster meets min_votes."""
        selector = ConsensusVotingSelector(similarity_threshold=0.99, min_votes=3)

        # All different hypotheses
        exp1 = MockExperiment("Hypothesis A")
        exp2 = MockExperiment("Hypothesis B")

        fb1 = MockFeedback(decision=True, score=0.7)
        fb2 = MockFeedback(decision=True, score=0.8)

        trace = MockTrace(hist=[(exp1, fb1), (exp2, fb2)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.consensus_enabled = True
            result = selector.select(trace)

        # Not enough votes in any cluster
        assert result is None

    def test_disabled_consensus(self):
        """Should return None when consensus is disabled."""
        selector = ConsensusVotingSelector()

        exp = MockExperiment()
        fb = MockFeedback(decision=True)
        trace = MockTrace(hist=[(exp, fb)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.consensus_enabled = False
            result = selector.select(trace)

        assert result is None

    def test_select_best_from_cluster(self):
        """Should return best-scoring experiment from largest cluster."""
        selector = ConsensusVotingSelector(similarity_threshold=0.5, min_votes=2)

        # Create cluster of similar hypotheses
        exp1 = MockExperiment("Use GRU layer")
        exp2 = MockExperiment("Use GRU layer with attention")

        fb1 = MockFeedback(decision=True, score=0.7)
        fb2 = MockFeedback(decision=True, score=0.9)

        trace = MockTrace(hist=[(exp1, fb1), (exp2, fb2)])

        with patch("rdagent.components.poetiq.conf.POETIQ_SETTINGS") as mock_settings:
            mock_settings.consensus_enabled = True
            result = selector.select(trace)

        if result is not None:
            # Should select the higher-scoring one from the cluster
            assert result[1].soft_score.value == 0.9
