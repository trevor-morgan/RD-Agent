"""AST-based static analysis for pandas pitfall detection.

This module provides pre-execution code linting to catch common pandas
anti-patterns before they cause silent failures at runtime.
"""

import ast
from collections.abc import Iterator

from rdagent.components.coder.pitfall_detector.patterns import (
    PANDAS_PITFALLS,
    LintResult,
    PitfallPattern,
    get_pitfall_by_id,
)


class PitfallLinter:
    """AST-based linter for detecting pandas coding pitfalls.

    This linter parses Python code and walks the AST to find known
    anti-patterns that can cause silent failures in factor coding.
    """

    def __init__(self, patterns: list[PitfallPattern] | None = None) -> None:
        """Initialize the linter with pitfall patterns.

        Args:
            patterns: Custom patterns to check. Defaults to PANDAS_PITFALLS.
        """
        self.patterns = patterns or PANDAS_PITFALLS

    def lint_code(self, code: str) -> list[LintResult]:
        """Parse and lint code for pitfall patterns.

        Args:
            code: Python source code to analyze.

        Returns:
            List of detected pitfall issues.
        """
        results: list[LintResult] = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If code doesn't parse, we can't lint it
            return results

        # Walk the AST and check each node
        for node in ast.walk(tree):
            results.extend(self._check_node(node, code))

        return results

    def _check_node(self, node: ast.AST, source: str) -> Iterator[LintResult]:
        """Check a single AST node for pitfall patterns.

        Args:
            node: AST node to check.
            source: Original source code for extracting snippets.

        Yields:
            LintResult for each detected pitfall.
        """
        # Check for pd.DataFrame(series, columns=[...]) pattern (PANDAS_001)
        if result := self._check_dataframe_constructor(node, source):
            yield result

        # Check for missing $ prefix in Qlib column access (PANDAS_004)
        if result := self._check_qlib_column_access(node, source):
            yield result

        # Check for inplace on view (PANDAS_002)
        if result := self._check_inplace_on_slice(node, source):
            yield result

    def _check_dataframe_constructor(self, node: ast.AST, source: str) -> LintResult | None:
        """Detect pd.DataFrame(series, columns=[...]) pattern.

        This pattern creates an empty DataFrame when the first argument is a
        MultiIndex Series because the constructor misaligns the index.
        """
        if not isinstance(node, ast.Call):
            return None

        # Check if this is pd.DataFrame(...) or DataFrame(...)
        func = node.func
        is_pd_dataframe = False

        if isinstance(func, ast.Attribute):
            # pd.DataFrame(...)
            if func.attr == "DataFrame" and isinstance(func.value, ast.Name) and func.value.id == "pd":
                is_pd_dataframe = True
        elif isinstance(func, ast.Name):
            # DataFrame(...) - assume pandas imported
            if func.id == "DataFrame":
                is_pd_dataframe = True

        if not is_pd_dataframe:
            return None

        # Check if 'columns' keyword is used
        has_columns_kwarg = any(kw.arg == "columns" for kw in node.keywords)

        if not has_columns_kwarg:
            return None

        # Check if first positional argument exists (potential series)
        if not node.args:
            return None

        first_arg = node.args[0]

        # Heuristics: if first arg is a variable name (not a dict/list literal),
        # it might be a Series. Flag it as suspicious.
        if isinstance(first_arg, ast.Name):
            # Variable being passed - could be a Series
            pitfall = get_pitfall_by_id("PANDAS_001")
            if pitfall:
                snippet = self._get_source_segment(source, node)
                var_name = first_arg.id
                return LintResult(
                    pitfall=pitfall,
                    line_number=node.lineno,
                    column=node.col_offset,
                    code_snippet=snippet,
                    suggested_fix=f'{var_name}.to_frame(name="<column_name>")',
                )

        return None

    def _check_qlib_column_access(self, node: ast.AST, source: str) -> LintResult | None:
        """Detect column access without $ prefix for Qlib data columns.

        Qlib uses $open, $close, $high, $low, $volume, $factor as column names.
        """
        qlib_columns = {"open", "close", "high", "low", "volume", "factor"}

        if not isinstance(node, ast.Subscript):
            return None

        # Check if subscript key is a string literal matching Qlib column (without $)
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            col_name = node.slice.value
            if col_name in qlib_columns:
                pitfall = get_pitfall_by_id("PANDAS_004")
                if pitfall:
                    snippet = self._get_source_segment(source, node)
                    return LintResult(
                        pitfall=pitfall,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=snippet,
                        suggested_fix=f'Use "${col_name}" instead of "{col_name}"',
                    )

        return None

    def _check_inplace_on_slice(self, node: ast.AST, source: str) -> LintResult | None:
        """Detect inplace=True operations on DataFrame slices/views."""
        if not isinstance(node, ast.Call):
            return None

        # Check if inplace=True is in keywords
        has_inplace_true = False
        for kw in node.keywords:
            if kw.arg == "inplace":
                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    has_inplace_true = True
                    break

        if not has_inplace_true:
            return None

        # Check if the call is on a subscript (df[...].method(...))
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Subscript):
                pitfall = get_pitfall_by_id("PANDAS_002")
                if pitfall:
                    snippet = self._get_source_segment(source, node)
                    return LintResult(
                        pitfall=pitfall,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=snippet,
                        suggested_fix="Avoid inplace=True on sliced DataFrames. Assign result instead.",
                    )

        return None

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Extract the source code segment for an AST node."""
        try:
            return ast.get_source_segment(source, node) or "<unknown>"
        except Exception:
            # Fallback for older Python versions or edge cases
            lines = source.split("\n")
            if hasattr(node, "lineno") and 0 < node.lineno <= len(lines):
                return lines[node.lineno - 1].strip()
            return "<unknown>"

    def get_critical_issues(self, code: str) -> list[LintResult]:
        """Get only critical severity issues from code.

        Args:
            code: Python source code to analyze.

        Returns:
            List of critical pitfall issues only.
        """
        all_results = self.lint_code(code)
        return [r for r in all_results if r.severity == "critical"]

    def format_results(self, results: list[LintResult]) -> str:
        """Format lint results as a human-readable report.

        Args:
            results: List of lint results to format.

        Returns:
            Formatted string report.
        """
        if not results:
            return "No pitfalls detected."

        lines = [f"Found {len(results)} potential issue(s):\n"]
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result}\n")

        return "\n".join(lines)


def lint_factor_code(code: str) -> list[LintResult]:
    """Convenience function to lint factor code for common pitfalls.

    Args:
        code: Factor implementation source code.

    Returns:
        List of detected pitfall issues.
    """
    linter = PitfallLinter()
    return linter.lint_code(code)


def has_critical_pitfalls(code: str) -> bool:
    """Quick check if code has any critical pitfalls.

    Args:
        code: Python source code to check.

    Returns:
        True if any critical pitfalls are detected.
    """
    linter = PitfallLinter()
    critical = linter.get_critical_issues(code)
    return len(critical) > 0
