"""Trajectory formatting for Poetiq.

Formats experiment history as structured context for LLM prompts,
helping the model understand the progression and learn from past attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdagent.core.proposal import Trace


@dataclass
class TrajectoryEntry:
    """Single entry in the trajectory history."""

    index: int
    hypothesis_text: str
    decision: bool
    score: float | None
    feedback_summary: str
    result_summary: str


class TrajectoryFormatter:
    """Format experiment history as structured context for LLM.

    Generates human-readable trajectory summaries that help the LLM
    understand what has been tried, what worked, and what didn't.
    """

    def __init__(
        self,
        max_entries: int = 10,
        include_failed: bool = True,
    ) -> None:
        """Initialize trajectory formatter.

        Args:
            max_entries: Maximum number of entries to include
            include_failed: Whether to include failed experiments
        """
        self.max_entries = max_entries
        self.include_failed = include_failed

    def format(self, trace: Trace) -> str:
        """Format trace as structured text for LLM context.

        Args:
            trace: Experiment trace history

        Returns:
            Formatted trajectory string
        """

        entries = self._extract_entries(trace)

        if not entries:
            return "## Experiment Trajectory\n\nNo experiments recorded yet."

        lines = ["## Experiment Trajectory\n"]

        # Add summary stats
        total = len(trace.hist)
        successes = sum(1 for e in entries if e.decision)
        lines.append(f"Total experiments: {total}, Successful: {successes}\n")

        # Add individual entries
        for entry in entries[-self.max_entries :]:
            lines.append(self._format_entry(entry))

        # Add overall summary
        lines.append(self._format_summary(entries))

        return "\n".join(lines)

    def format_for_hypothesis_gen(self, trace: Trace) -> dict[str, str]:
        """Format trace specifically for hypothesis generation prompts.

        Args:
            trace: Experiment trace history

        Returns:
            Dict with trajectory, top_experiments, and failure_patterns
        """
        entries = self._extract_entries(trace)

        return {
            "trajectory": self.format(trace),
            "top_experiments": self._format_top_experiments(entries),
            "failure_patterns": self._format_failures(entries),
            "improvement_trend": self._format_trend(entries),
        }

    def _extract_entries(self, trace: Trace) -> list[TrajectoryEntry]:
        """Extract trajectory entries from trace.

        Args:
            trace: Experiment trace history

        Returns:
            List of TrajectoryEntry objects
        """
        entries: list[TrajectoryEntry] = []

        for i, (exp, fb) in enumerate(trace.hist):
            if not self.include_failed and not fb.decision:
                continue

            # Extract hypothesis text
            hypothesis_text = ""
            if exp.hypothesis is not None:
                hypothesis_text = exp.hypothesis.hypothesis[:200]

            # Extract score
            score: float | None = None
            soft_score = getattr(fb, "soft_score", None)
            if soft_score is not None:
                score = soft_score.value
            elif fb.decision:
                score = 1.0
            else:
                score = 0.0

            # Extract feedback summary
            feedback_summary = ""
            if hasattr(fb, "observations"):
                feedback_summary = fb.observations[:150] if fb.observations else ""

            # Extract result summary
            result_summary = ""
            if exp.result is not None:
                if hasattr(exp.result, "to_dict"):
                    result_summary = str(exp.result.to_dict())[:100]
                else:
                    result_summary = str(exp.result)[:100]

            entries.append(
                TrajectoryEntry(
                    index=i,
                    hypothesis_text=hypothesis_text,
                    decision=fb.decision,
                    score=score,
                    feedback_summary=feedback_summary,
                    result_summary=result_summary,
                )
            )

        return entries

    def _format_entry(self, entry: TrajectoryEntry) -> str:
        """Format single trajectory entry.

        Args:
            entry: TrajectoryEntry to format

        Returns:
            Formatted string
        """
        status = "[SUCCESS]" if entry.decision else "[BELOW_THRESHOLD]"
        score_str = f"{entry.score:.2f}" if entry.score is not None else "N/A"

        return f"""
### Experiment {entry.index + 1} {status}
- **Hypothesis**: {entry.hypothesis_text}...
- **Score**: {score_str}
- **Result**: {entry.result_summary}
- **Feedback**: {entry.feedback_summary}...
"""

    def _format_summary(self, entries: list[TrajectoryEntry]) -> str:
        """Format overall trajectory summary.

        Args:
            entries: List of trajectory entries

        Returns:
            Summary string
        """
        if not entries:
            return "\n### Summary\nNo experiments to summarize."

        successes = [e for e in entries if e.decision]
        failures = [e for e in entries if not e.decision]

        scores = [e.score for e in entries if e.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return f"""
### Summary
- Successful experiments: {len(successes)}
- Failed experiments: {len(failures)}
- Average score: {avg_score:.2f}
- Best score: {max(scores) if scores else 'N/A'}
"""

    def _format_top_experiments(self, entries: list[TrajectoryEntry], k: int = 3) -> str:
        """Format top-K experiments by score.

        Args:
            entries: List of trajectory entries
            k: Number of top experiments to include

        Returns:
            Top experiments summary string
        """
        if not entries:
            return "\n### Top Experiments\nNo experiments recorded yet."

        # Sort by score descending
        scored_entries = [(e, e.score or 0.0) for e in entries]
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        top_k = scored_entries[:k]

        lines = ["\n### Top Experiments"]
        for i, (entry, score) in enumerate(top_k, 1):
            lines.append(
                f"- **Rank {i}** (Score: {score:.2f}): {entry.hypothesis_text[:100]}..."
            )

        return "\n".join(lines)

    def _format_failures(self, entries: list[TrajectoryEntry]) -> str:
        """Format common failure patterns.

        Args:
            entries: List of trajectory entries

        Returns:
            Failure patterns string
        """
        failures = [e for e in entries if not e.decision]

        if not failures:
            return "\n### Failure Patterns\nNo failures to analyze."

        # Get recent failures
        recent_failures = failures[-3:]

        lines = ["\n### Recent Failures"]
        for f in recent_failures:
            lines.append(f"- Exp {f.index + 1}: {f.hypothesis_text[:100]}...")

        return "\n".join(lines)

    def _format_trend(self, entries: list[TrajectoryEntry]) -> str:
        """Format improvement trend analysis.

        Args:
            entries: List of trajectory entries

        Returns:
            Trend analysis string
        """
        scores = [e.score for e in entries if e.score is not None]

        if len(scores) < 2:
            return "\n### Trend\nInsufficient data for trend analysis."

        # Simple trend: compare recent vs earlier
        mid = len(scores) // 2
        early_avg = sum(scores[:mid]) / mid if mid > 0 else 0
        late_avg = sum(scores[mid:]) / (len(scores) - mid) if len(scores) > mid else 0

        if late_avg > early_avg * 1.1:
            trend = "IMPROVING"
        elif late_avg < early_avg * 0.9:
            trend = "DECLINING"
        else:
            trend = "STABLE"

        return f"""
### Trend Analysis
- Early average score: {early_avg:.2f}
- Recent average score: {late_avg:.2f}
- Trend: {trend}
"""
