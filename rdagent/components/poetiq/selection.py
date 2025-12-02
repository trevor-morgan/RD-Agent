"""SOTA selection strategies for Poetiq.

Provides stochastic and consensus-based selection of experiments,
replacing the deterministic "most recent decision=True" approach.
"""

from __future__ import annotations

import math
import random
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.core.experiment import Experiment
    from rdagent.core.proposal import ExperimentFeedback, Trace

from rdagent.components.poetiq.feedback import SoftScore


class StochasticSOTASelector:
    """Sample from top-K experiments instead of deterministic best.

    Supports uniform and softmax sampling strategies to balance
    exploitation (selecting proven approaches) vs exploration
    (giving other good approaches a chance).
    """

    def __init__(
        self,
        k: int = 3,
        temperature: float = 1.0,
        sampling: str = "softmax",
    ) -> None:
        """Initialize stochastic selector.

        Args:
            k: Number of top experiments to consider
            temperature: Softmax temperature (higher = more uniform)
            sampling: Strategy - 'uniform' or 'softmax'
        """
        self.k = k
        self.temperature = temperature
        self.sampling = sampling

    def get_top_k(
        self,
        trace: Trace,
    ) -> list[tuple[int, Experiment, ExperimentFeedback, float]]:
        """Get top-K experiments with their scores.

        Args:
            trace: Experiment trace history

        Returns:
            List of (index, experiment, feedback, score) tuples sorted by score desc
        """
        candidates: list[tuple[int, Experiment, ExperimentFeedback, float]] = []

        for i, (exp, fb) in enumerate(trace.hist):
            if not fb.decision:
                continue

            # Extract soft score if available
            soft_score = getattr(fb, "soft_score", None)
            if soft_score is not None:
                score_val = soft_score.value
            else:
                score_val = 1.0  # Default for binary decision=True

            candidates.append((i, exp, fb, score_val))

        # Sort by score descending
        candidates.sort(key=lambda x: x[3], reverse=True)
        return candidates[: self.k]

    def select(
        self,
        trace: Trace,
    ) -> tuple[Experiment, ExperimentFeedback] | None:
        """Select SOTA experiment using configured sampling strategy.

        Args:
            trace: Experiment trace history

        Returns:
            (experiment, feedback) tuple or None if no candidates
        """
        from rdagent.components.poetiq.conf import POETIQ_SETTINGS

        # Fall back to original behavior if disabled or k=1
        if not POETIQ_SETTINGS.enabled or self.k <= 1:
            return self._select_best(trace)

        candidates = self.get_top_k(trace)
        if not candidates:
            return None

        if self.sampling == "uniform":
            _, exp, fb, _ = random.choice(candidates)
        else:  # softmax
            _, exp, fb, _ = self._softmax_sample(candidates)

        return exp, fb

    def _softmax_sample(
        self,
        candidates: list[tuple[int, Experiment, ExperimentFeedback, float]],
    ) -> tuple[int, Experiment, ExperimentFeedback, float]:
        """Sample from candidates using softmax distribution.

        Args:
            candidates: List of (index, experiment, feedback, score) tuples

        Returns:
            Selected candidate tuple
        """
        scores = [c[3] for c in candidates]

        # Apply temperature scaling
        exp_scores = [math.exp(s / self.temperature) for s in scores]
        total = sum(exp_scores)

        if total == 0:
            return random.choice(candidates)

        probs = [e / total for e in exp_scores]

        # Sample based on probabilities
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return candidates[i]

        return candidates[-1]

    def _select_best(
        self,
        trace: Trace,
    ) -> tuple[Experiment, ExperimentFeedback] | None:
        """Original deterministic selection - most recent decision=True.

        Args:
            trace: Experiment trace history

        Returns:
            (experiment, feedback) tuple or None
        """
        for exp, fb in reversed(trace.hist):
            if fb.decision:
                return exp, fb
        return None


class ConsensusVotingSelector:
    """Cluster similar experiments and vote on robustness.

    Groups experiments by hypothesis similarity, then selects
    from the largest cluster - indicating multiple independent
    runs converged on similar solutions.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_votes: int = 2,
    ) -> None:
        """Initialize consensus selector.

        Args:
            similarity_threshold: Minimum similarity to cluster together (0.0-1.0)
            min_votes: Minimum cluster size to consider for selection
        """
        self.similarity_threshold = similarity_threshold
        self.min_votes = min_votes

    def select(
        self,
        trace: Trace,
    ) -> tuple[Experiment, ExperimentFeedback] | None:
        """Select SOTA experiment using consensus voting.

        Args:
            trace: Experiment trace history

        Returns:
            (experiment, feedback) tuple or None if insufficient consensus
        """
        from rdagent.components.poetiq.conf import POETIQ_SETTINGS

        if not POETIQ_SETTINGS.consensus_enabled:
            return None

        # Collect successful candidates
        candidates: list[tuple[int, Experiment, ExperimentFeedback]] = [
            (i, exp, fb) for i, (exp, fb) in enumerate(trace.hist) if fb.decision
        ]

        if len(candidates) < self.min_votes:
            return None

        # Cluster by similarity
        clusters = self._cluster_by_similarity(candidates)

        if not clusters:
            return None

        # Find largest cluster
        largest = max(clusters, key=len)

        if len(largest) < self.min_votes:
            return None

        # Return best from largest cluster (by soft score)
        best = max(
            largest,
            key=lambda x: getattr(x[2], "soft_score", SoftScore(0.0)).value,
        )

        return best[1], best[2]

    def _cluster_by_similarity(
        self,
        candidates: list[tuple[int, Experiment, ExperimentFeedback]],
    ) -> list[list[tuple[int, Experiment, ExperimentFeedback]]]:
        """Cluster candidates by hypothesis text similarity.

        Uses union-find style greedy clustering based on text similarity.

        Args:
            candidates: List of (index, experiment, feedback) tuples

        Returns:
            List of clusters, each a list of candidate tuples
        """
        n = len(candidates)
        assigned = [False] * n
        clusters: list[list[tuple[int, Experiment, ExperimentFeedback]]] = []

        for i in range(n):
            if assigned[i]:
                continue

            cluster = [candidates[i]]
            assigned[i] = True

            # Get hypothesis text for comparison
            h1 = self._get_hypothesis_text(candidates[i][1])

            for j in range(i + 1, n):
                if assigned[j]:
                    continue

                h2 = self._get_hypothesis_text(candidates[j][1])

                # Check similarity
                similarity = SequenceMatcher(None, h1, h2).ratio()
                if similarity >= self.similarity_threshold:
                    cluster.append(candidates[j])
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _get_hypothesis_text(self, exp: Experiment) -> str:
        """Extract hypothesis text from experiment.

        Args:
            exp: Experiment object

        Returns:
            Hypothesis text string
        """
        if exp.hypothesis is not None:
            return exp.hypothesis.hypothesis
        return ""

    def get_cluster_info(
        self,
        trace: Trace,
    ) -> dict[str, int | list[int]]:
        """Get information about clustering results.

        Args:
            trace: Experiment trace history

        Returns:
            Dict with cluster sizes and indices
        """
        candidates = [(i, exp, fb) for i, (exp, fb) in enumerate(trace.hist) if fb.decision]

        if not candidates:
            return {"num_clusters": 0, "cluster_sizes": [], "largest_cluster_indices": []}

        clusters = self._cluster_by_similarity(candidates)

        return {
            "num_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "largest_cluster_indices": [c[0] for c in max(clusters, key=len)] if clusters else [],
        }
