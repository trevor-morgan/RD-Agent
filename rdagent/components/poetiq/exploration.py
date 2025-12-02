"""Parallel expert exploration for Poetiq.

Generates multiple hypotheses with different seeds/configurations,
similar to Poetiq's multi-expert ensemble approach.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.core.proposal import Hypothesis, HypothesisGen, Trace


class ParallelHypothesisGen:
    """Wrapper that generates multiple hypotheses with different seeds.

    Enables exploration diversity by running the same hypothesis generator
    multiple times with different random seeds, similar to Poetiq's
    parallel expert approach.
    """

    def __init__(self, base_gen: HypothesisGen) -> None:
        """Initialize parallel generator.

        Args:
            base_gen: Base hypothesis generator to wrap
        """
        self.base_gen = base_gen
        # Serialize seed overrides to avoid cross-task leakage when multiple
        # hypotheses are generated in parallel.
        self._seed_lock = threading.Lock()

    def gen(self, trace: Trace, plan: object = None) -> Hypothesis:
        """Generate single hypothesis (backward compatible).

        Args:
            trace: Experiment trace history
            plan: Optional plan object

        Returns:
            Single hypothesis from base generator
        """
        return self.base_gen.gen(trace, plan)

    def gen_parallel(self, trace: Trace, plan: object = None) -> list[Hypothesis]:
        """Generate multiple hypotheses with different seeds.

        Each hypothesis is generated with a different LLM cache seed,
        providing diversity in the generated approaches.

        Args:
            trace: Experiment trace history
            plan: Optional plan object

        Returns:
            List of hypotheses (may contain duplicates if LLM returns similar results)
        """
        from rdagent.components.poetiq.conf import POETIQ_SETTINGS

        n = POETIQ_SETTINGS.parallel_experts
        if n <= 1:
            return [self.gen(trace, plan)]

        from rdagent.oai.llm_conf import LLM_SETTINGS

        hypotheses: list[Hypothesis] = []
        original_seed = LLM_SETTINGS.init_chat_cache_seed

        for i in range(n):
            try:
                with self._seed_lock:
                    # Modify seed for diversity
                    LLM_SETTINGS.init_chat_cache_seed = (
                        original_seed + i * POETIQ_SETTINGS.parallel_expert_seed_offset
                    )

                    h = self.base_gen.gen(trace, plan)
                if h is not None:
                    hypotheses.append(h)

            except Exception:
                # Continue with other experts if one fails
                pass
            finally:
                # Restore original seed
                with self._seed_lock:
                    LLM_SETTINGS.init_chat_cache_seed = original_seed

        # Ensure we return at least one hypothesis
        if not hypotheses:
            fallback = self.gen(trace, plan)
            if fallback is not None:
                hypotheses.append(fallback)

        return hypotheses

    def gen_with_seed(
        self,
        trace: Trace,
        seed_offset: int,
        plan: object = None,
    ) -> Hypothesis | None:
        """Generate hypothesis with specific seed offset.

        Args:
            trace: Experiment trace history
            seed_offset: Offset to add to base seed
            plan: Optional plan object

        Returns:
            Hypothesis or None if generation fails
        """
        from rdagent.oai.llm_conf import LLM_SETTINGS

        original_seed = LLM_SETTINGS.init_chat_cache_seed

        try:
            LLM_SETTINGS.init_chat_cache_seed = original_seed + seed_offset
            return self.base_gen.gen(trace, plan)
        finally:
            LLM_SETTINGS.init_chat_cache_seed = original_seed


def select_diverse_hypotheses(
    hypotheses: list[Hypothesis],
    max_select: int = 3,
    similarity_threshold: float = 0.7,
) -> list[Hypothesis]:
    """Select diverse hypotheses from a pool.

    Filters out similar hypotheses to maximize exploration diversity.

    Args:
        hypotheses: Pool of candidate hypotheses
        max_select: Maximum number to select
        similarity_threshold: Threshold below which hypotheses are considered different

    Returns:
        List of diverse hypotheses
    """
    from difflib import SequenceMatcher

    if len(hypotheses) <= max_select:
        return hypotheses

    selected: list[Hypothesis] = []

    for h in hypotheses:
        is_diverse = True
        h_text = h.hypothesis

        for s in selected:
            similarity = SequenceMatcher(None, h_text, s.hypothesis).ratio()
            if similarity >= similarity_threshold:
                is_diverse = False
                break

        if is_diverse:
            selected.append(h)
            if len(selected) >= max_select:
                break

    return selected
