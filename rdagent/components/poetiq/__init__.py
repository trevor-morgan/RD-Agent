"""Poetiq strategies for enhanced agent exploration.

This module implements strategies inspired by the Poetiq ARC-AGI solver:
- Soft Scoring: Continuous scores instead of binary decisions
- Stochastic SOTA Selection: Sample from top-K experiments
- Consensus Voting: Group similar experiments and vote
- Parallel Expert Exploration: Generate multiple hypotheses with different seeds
- Early Exit: Stop when metrics exceed threshold
- Trajectory Formatting: Structured history for LLM context
"""

from rdagent.components.poetiq.conf import POETIQ_SETTINGS, PoetiqSettings
from rdagent.components.poetiq.early_exit import EarlyExitChecker
from rdagent.components.poetiq.exploration import ParallelHypothesisGen
from rdagent.components.poetiq.feedback import (
    ScoredHypothesisFeedback,
    SoftScore,
    compute_soft_score,
)
from rdagent.components.poetiq.selection import (
    ConsensusVotingSelector,
    StochasticSOTASelector,
)
from rdagent.components.poetiq.trajectory import TrajectoryFormatter

__all__ = [
    "POETIQ_SETTINGS",
    "ConsensusVotingSelector",
    "EarlyExitChecker",
    "ParallelHypothesisGen",
    "PoetiqSettings",
    "ScoredHypothesisFeedback",
    "SoftScore",
    "StochasticSOTASelector",
    "TrajectoryFormatter",
    "compute_soft_score",
]
