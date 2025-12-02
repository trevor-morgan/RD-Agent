"""Configuration for Poetiq exploration strategies.

All features are disabled by default for backward compatibility.
Enable via environment variables with POETIQ_ prefix.
"""

from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from rdagent.core.conf import ExtendedBaseSettings


class PoetiqSettings(ExtendedBaseSettings):
    """Configuration for Poetiq exploration strategies."""

    model_config = SettingsConfigDict(env_prefix="POETIQ_", protected_namespaces=())

    # Master switch
    enabled: bool = False
    """Enable Poetiq features globally. When False, all features are disabled."""

    # Parallel Expert Exploration
    parallel_experts: int = 1
    """Number of parallel hypothesis generators. 1 = disabled, >1 = parallel."""

    parallel_expert_seed_offset: int = 1000
    """Seed offset between parallel experts for diversity."""

    # Soft Scoring
    soft_scoring: bool = False
    """Enable soft scoring (continuous 0.0-1.0) instead of binary decisions."""

    score_threshold: float = 0.5
    """Threshold for converting soft score to binary decision."""

    score_metric: str = "IC"
    """Primary metric for computing soft scores."""

    # Stochastic SOTA Selection
    stochastic_sota_k: int = 1
    """Top-K experiments to consider for SOTA selection. 1 = deterministic (current behavior)."""

    stochastic_sota_temperature: float = 1.0
    """Temperature for softmax sampling. Higher = more uniform, Lower = more greedy."""

    stochastic_sampling: str = "softmax"
    """Sampling strategy: 'uniform' or 'softmax'."""

    # Voting/Consensus
    consensus_enabled: bool = False
    """Enable consensus voting for robustness."""

    consensus_similarity_threshold: float = 0.8
    """Similarity threshold for clustering hypotheses (0.0-1.0)."""

    consensus_min_votes: int = 2
    """Minimum cluster size to consider for consensus."""

    # Trajectory Context
    trajectory_depth: int = 10
    """Maximum number of experiments to include in trajectory context."""

    trajectory_include_failed: bool = True
    """Include failed experiments in trajectory context."""

    trajectory_format: str = "structured"
    """Format for trajectory: 'structured' or 'simple'."""

    # Early Exit
    early_exit_metric: str | None = None
    """Metric to check for early exit. None = disabled."""

    early_exit_threshold: float | None = None
    """Threshold value for early exit. None = disabled."""

    early_exit_direction: str = "higher"
    """Direction for threshold comparison: 'higher' or 'lower'."""


POETIQ_SETTINGS = PoetiqSettings()
