"""Knowledge bridge for pitfall patterns.

This module provides utilities to integrate pitfall detection with the
CoSTEER knowledge management system, enabling RAG-based retrieval of
similar pitfalls and their solutions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rdagent.components.coder.pitfall_detector.linter import PitfallLinter
from rdagent.components.coder.pitfall_detector.patterns import (
    PANDAS_PITFALLS,
    get_pitfall_by_id,
)


@dataclass
class PitfallKnowledgeEntry:
    """A knowledge entry for a pitfall.

    Attributes:
        pitfall_id: ID of the pitfall pattern
        example_bad_code: Example of code with the pitfall
        example_good_code: Corrected version of the code
        error_context: Error messages or symptoms that led to detection
        resolution_notes: Additional notes on how to resolve
    """

    pitfall_id: str
    example_bad_code: str
    example_good_code: str
    error_context: str = ""
    resolution_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "pitfall_id": self.pitfall_id,
            "example_bad_code": self.example_bad_code,
            "example_good_code": self.example_good_code,
            "error_context": self.error_context,
            "resolution_notes": self.resolution_notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PitfallKnowledgeEntry":
        return cls(**data)


class PitfallKnowledgeBridge:
    """Bridge between pitfall detection and knowledge management.

    This class maintains a local knowledge base of pitfall examples and
    provides methods to query for similar pitfalls based on code patterns.
    """

    def __init__(self, knowledge_path: Path | None = None) -> None:
        """Initialize the knowledge bridge.

        Args:
            knowledge_path: Optional path to persist knowledge entries.
                           If None, uses in-memory storage only.
        """
        self.knowledge_path = knowledge_path
        self.entries: list[PitfallKnowledgeEntry] = []
        self._load_builtin_knowledge()

        if knowledge_path and knowledge_path.exists():
            self._load_from_file()

    def _load_builtin_knowledge(self) -> None:
        """Load built-in knowledge from pitfall patterns."""
        for pattern in PANDAS_PITFALLS:
            entry = PitfallKnowledgeEntry(
                pitfall_id=pattern.id,
                example_bad_code=pattern.bad_example,
                example_good_code=pattern.good_example,
                error_context=pattern.description,
                resolution_notes=pattern.correction_template,
            )
            self.entries.append(entry)

    def _load_from_file(self) -> None:
        """Load additional knowledge entries from file."""
        if not self.knowledge_path or not self.knowledge_path.exists():
            return

        try:
            with self.knowledge_path.open() as f:
                data = json.load(f)
                for item in data.get("entries", []):
                    self.entries.append(PitfallKnowledgeEntry.from_dict(item))
        except (json.JSONDecodeError, OSError):
            pass  # Ignore corrupted files

    def _save_to_file(self) -> None:
        """Save knowledge entries to file."""
        if not self.knowledge_path:
            return

        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        with self.knowledge_path.open("w") as f:
            data = {
                "entries": [e.to_dict() for e in self.entries if e.pitfall_id.startswith("CUSTOM_")],
            }
            json.dump(data, f, indent=2)

    def register_pitfall(
        self,
        pitfall_id: str,
        bad_code: str,
        good_code: str,
        error_context: str = "",
        notes: str = "",
    ) -> None:
        """Register a new pitfall example.

        Args:
            pitfall_id: Unique ID for this pitfall (use CUSTOM_ prefix for user-defined)
            bad_code: Example of code with the pitfall
            good_code: Corrected version
            error_context: Error messages or symptoms
            notes: Additional resolution notes
        """
        entry = PitfallKnowledgeEntry(
            pitfall_id=pitfall_id,
            example_bad_code=bad_code,
            example_good_code=good_code,
            error_context=error_context,
            resolution_notes=notes,
        )
        self.entries.append(entry)
        self._save_to_file()

    def query_similar_pitfalls(self, code: str, limit: int = 3) -> list[PitfallKnowledgeEntry]:
        """Find pitfall entries similar to patterns in the given code.

        Args:
            code: Source code to analyze.
            limit: Maximum number of entries to return.

        Returns:
            List of relevant pitfall knowledge entries.
        """
        linter = PitfallLinter()
        lint_results = linter.lint_code(code)

        # Get pitfall IDs detected in the code
        detected_ids = {r.pitfall.id for r in lint_results}

        # Return entries matching detected pitfalls
        relevant = []
        for entry in self.entries:
            if entry.pitfall_id in detected_ids:
                relevant.append(entry)
                if len(relevant) >= limit:
                    break

        return relevant

    def get_correction_context(self, pitfall_id: str) -> str:
        """Get context for prompt enhancement based on a pitfall.

        Args:
            pitfall_id: ID of the pitfall.

        Returns:
            Formatted context string for LLM prompts.
        """
        entries = [e for e in self.entries if e.pitfall_id == pitfall_id]
        pattern = get_pitfall_by_id(pitfall_id)

        if not entries and not pattern:
            return ""

        lines = [f"=== Pitfall Knowledge: {pitfall_id} ==="]

        if pattern:
            lines.extend([
                f"Name: {pattern.name}",
                f"Severity: {pattern.severity}",
                f"Description: {pattern.description}",
                "",
            ])

        for i, entry in enumerate(entries[:3], 1):
            lines.extend([
                f"Example {i}:",
                f"  Bad code:  {entry.example_bad_code}",
                f"  Good code: {entry.example_good_code}",
                "",
            ])

        if pattern:
            lines.append(f"Fix: {pattern.correction_template}")

        return "\n".join(lines)

    def format_prompt_context(self, code: str) -> str:
        """Generate prompt context for code that may have pitfalls.

        Args:
            code: Source code to analyze.

        Returns:
            Formatted context to append to LLM prompts.
        """
        similar = self.query_similar_pitfalls(code)

        if not similar:
            return ""

        lines = ["\n=== Detected Code Pitfalls ==="]
        lines.append("Your previous attempt may have failed due to common pitfalls. Here's relevant knowledge:\n")

        for entry in similar:
            context = self.get_correction_context(entry.pitfall_id)
            if context:
                lines.append(context)
                lines.append("")

        return "\n".join(lines)


# Global singleton for easy access
_default_bridge: PitfallKnowledgeBridge | None = None


def get_pitfall_knowledge_bridge() -> PitfallKnowledgeBridge:
    """Get the global pitfall knowledge bridge instance."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = PitfallKnowledgeBridge()
    return _default_bridge


def get_pitfall_prompt_context(code: str) -> str:
    """Convenience function to get prompt context for code.

    Args:
        code: Source code to analyze.

    Returns:
        Formatted context for LLM prompts.
    """
    bridge = get_pitfall_knowledge_bridge()
    return bridge.format_prompt_context(code)
