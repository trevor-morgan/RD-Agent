"""Auto-correction for pandas pitfall patterns.

This module provides code transformation utilities to automatically fix
known pandas anti-patterns in factor implementations.
"""

import re
from dataclasses import dataclass

from rdagent.components.coder.pitfall_detector.linter import PitfallLinter
from rdagent.components.coder.pitfall_detector.patterns import LintResult


@dataclass
class CodeCorrection:
    """A suggested code correction.

    Attributes:
        original: The original problematic code
        corrected: The suggested corrected code
        pitfall_id: ID of the pitfall being corrected
        line_number: Line number of the original code
        confidence: How confident we are in this correction (0.0-1.0)
        explanation: Human-readable explanation of the change
    """

    original: str
    corrected: str
    pitfall_id: str
    line_number: int
    confidence: float
    explanation: str


class PitfallCorrector:
    """Automated code corrector for pandas pitfalls.

    Generates corrected code snippets for detected pitfalls and can
    optionally apply corrections to the full source code.
    """

    def __init__(self) -> None:
        self.linter = PitfallLinter()

    def suggest_corrections(self, code: str) -> list[CodeCorrection]:
        """Generate correction suggestions for all detected pitfalls.

        Args:
            code: Python source code to analyze.

        Returns:
            List of suggested corrections.
        """
        lint_results = self.linter.lint_code(code)
        corrections = []

        for result in lint_results:
            correction = self._generate_correction(result, code)
            if correction:
                corrections.append(correction)

        return corrections

    def _generate_correction(self, lint_result: LintResult, full_code: str) -> CodeCorrection | None:
        """Generate a correction for a single lint result.

        Args:
            lint_result: The detected pitfall.
            full_code: Full source code for context.

        Returns:
            CodeCorrection or None if no correction can be generated.
        """
        if lint_result.pitfall.id == "PANDAS_001":
            return self._correct_dataframe_constructor(lint_result, full_code)
        if lint_result.pitfall.id == "PANDAS_004":
            return self._correct_qlib_column_access(lint_result, full_code)
        if lint_result.pitfall.id == "PANDAS_002":
            return self._correct_inplace_on_slice(lint_result, full_code)

        return None

    def _correct_dataframe_constructor(self, lint_result: LintResult, full_code: str) -> CodeCorrection | None:
        """Correct pd.DataFrame(series, columns=[...]) to series.to_frame().

        Example:
            Before: result = pd.DataFrame(momentum, columns=["Momentum"])
            After:  result = momentum.to_frame(name="Momentum")
        """
        snippet = lint_result.code_snippet

        # Try to extract the series variable name and column name
        # Pattern: pd.DataFrame(var, columns=["name"]) or pd.DataFrame(var, columns=['name'])
        pattern = r"pd\.DataFrame\s*\(\s*(\w+)\s*,\s*columns\s*=\s*\[(['\"])(.*?)\2\]\s*\)"
        match = re.search(pattern, snippet)

        if match:
            series_var = match.group(1)
            column_name = match.group(3)
            corrected = f'{series_var}.to_frame(name="{column_name}")'

            return CodeCorrection(
                original=snippet,
                corrected=corrected,
                pitfall_id="PANDAS_001",
                line_number=lint_result.line_number,
                confidence=0.9,
                explanation=(
                    f"Replace pd.DataFrame({series_var}, columns=[...]) with "
                    f"{series_var}.to_frame() to preserve MultiIndex and data."
                ),
            )

        return None

    def _correct_qlib_column_access(self, lint_result: LintResult, full_code: str) -> CodeCorrection | None:
        """Correct missing $ prefix in Qlib column access.

        Example:
            Before: df["close"]
            After:  df["$close"]
        """
        snippet = lint_result.code_snippet

        # Find the column name without $ prefix
        qlib_columns = {"open", "close", "high", "low", "volume", "factor"}

        for col in qlib_columns:
            pattern = rf'\[(["\']){col}\1\]'
            if re.search(pattern, snippet):
                corrected = re.sub(pattern, rf"[\1${col}\1]", snippet)
                return CodeCorrection(
                    original=snippet,
                    corrected=corrected,
                    pitfall_id="PANDAS_004",
                    line_number=lint_result.line_number,
                    confidence=0.95,
                    explanation=f'Qlib data uses "${col}" not "{col}" for column names.',
                )

        return None

    def _correct_inplace_on_slice(self, lint_result: LintResult, full_code: str) -> CodeCorrection | None:
        """Suggest removing inplace=True from sliced DataFrame operations.

        Example:
            Before: df[mask].fillna(0, inplace=True)
            After:  df.loc[mask] = df.loc[mask].fillna(0)
        """
        # This is more complex to auto-correct safely, so we just provide guidance
        return CodeCorrection(
            original=lint_result.code_snippet,
            corrected="# Avoid inplace=True on slices. Assign result instead.",
            pitfall_id="PANDAS_002",
            line_number=lint_result.line_number,
            confidence=0.6,
            explanation=(
                "inplace=True on DataFrame slices may not work as expected. "
                "Use df.loc[...] = df.loc[...].method() instead."
            ),
        )

    def apply_correction(self, code: str, correction: CodeCorrection) -> str:
        """Apply a single correction to the source code.

        Args:
            code: Original source code.
            correction: Correction to apply.

        Returns:
            Modified source code with the correction applied.
        """
        lines = code.split("\n")

        # Find and replace the problematic line
        if 0 < correction.line_number <= len(lines):
            line = lines[correction.line_number - 1]
            if correction.original in line:
                lines[correction.line_number - 1] = line.replace(
                    correction.original, correction.corrected
                )

        return "\n".join(lines)

    def apply_all_corrections(self, code: str) -> tuple[str, list[CodeCorrection]]:
        """Apply all possible corrections to the source code.

        Args:
            code: Original source code.

        Returns:
            Tuple of (corrected code, list of applied corrections).
        """
        corrections = self.suggest_corrections(code)
        applied = []

        # Apply corrections in reverse line order to preserve line numbers
        sorted_corrections = sorted(corrections, key=lambda c: c.line_number, reverse=True)

        for correction in sorted_corrections:
            if correction.confidence >= 0.8:  # Only apply high-confidence corrections
                code = self.apply_correction(code, correction)
                applied.append(correction)

        return code, applied

    def format_corrections(self, corrections: list[CodeCorrection]) -> str:
        """Format corrections as a human-readable report.

        Args:
            corrections: List of corrections to format.

        Returns:
            Formatted string report.
        """
        if not corrections:
            return "No corrections suggested."

        lines = [f"Found {len(corrections)} potential correction(s):\n"]
        for i, correction in enumerate(corrections, 1):
            lines.extend([
                f"{i}. [{correction.pitfall_id}] Line {correction.line_number} (confidence: {correction.confidence:.0%})",
                f"   Original:  {correction.original}",
                f"   Corrected: {correction.corrected}",
                f"   Reason:    {correction.explanation}",
                "",
            ])

        return "\n".join(lines)


def suggest_factor_corrections(code: str) -> list[CodeCorrection]:
    """Convenience function to get correction suggestions for factor code.

    Args:
        code: Factor implementation source code.

    Returns:
        List of suggested corrections.
    """
    corrector = PitfallCorrector()
    return corrector.suggest_corrections(code)


def auto_correct_factor_code(code: str) -> tuple[str, list[CodeCorrection]]:
    """Convenience function to auto-correct factor code.

    Only applies high-confidence corrections (>=80%).

    Args:
        code: Factor implementation source code.

    Returns:
        Tuple of (corrected code, list of applied corrections).
    """
    corrector = PitfallCorrector()
    return corrector.apply_all_corrections(code)
