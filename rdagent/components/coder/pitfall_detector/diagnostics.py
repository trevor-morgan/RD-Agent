"""Runtime diagnostics for pandas DataFrame issues.

This module provides post-execution analysis to diagnose why a DataFrame
might be empty or contain unexpected values, and links issues to known
pitfall patterns.
"""

import re
from dataclasses import dataclass, field

import pandas as pd
from rdagent.components.coder.pitfall_detector.linter import PitfallLinter
from rdagent.components.coder.pitfall_detector.patterns import Diagnosis, get_pitfall_by_id


@dataclass
class DataFrameAnalysis:
    """Analysis results for a DataFrame.

    Attributes:
        is_empty: Whether the DataFrame has no rows
        is_all_nan: Whether all values are NaN
        row_count: Number of rows
        nan_ratio: Ratio of NaN values (0.0 to 1.0)
        index_type: Type of index (e.g., "MultiIndex", "RangeIndex")
        column_count: Number of columns
        dtypes: Data types of columns
    """

    is_empty: bool
    is_all_nan: bool
    row_count: int
    nan_ratio: float
    index_type: str
    column_count: int
    dtypes: dict = field(default_factory=dict)


class RuntimeDiagnostics:
    """Post-execution analyzer for DataFrame issues.

    Analyzes DataFrame outputs and source code to determine why
    factor calculations might have failed silently.
    """

    def __init__(self) -> None:
        self.linter = PitfallLinter()

    def analyze_dataframe(self, df: pd.DataFrame) -> DataFrameAnalysis:
        """Analyze a DataFrame's structure and content.

        Args:
            df: DataFrame to analyze.

        Returns:
            Analysis results.
        """
        is_empty = len(df) == 0
        is_all_nan = df.isna().all().all() if not is_empty else True

        nan_count = df.isna().sum().sum() if not is_empty else 0
        total_cells = df.size if df.size > 0 else 1
        nan_ratio = nan_count / total_cells

        index_type = type(df.index).__name__

        return DataFrameAnalysis(
            is_empty=is_empty,
            is_all_nan=is_all_nan,
            row_count=len(df),
            nan_ratio=nan_ratio,
            index_type=index_type,
            column_count=len(df.columns),
            dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        )

    def diagnose_empty_dataframe(self, result_df: pd.DataFrame, code: str) -> Diagnosis:
        """Diagnose why a DataFrame result is empty.

        Args:
            result_df: The empty or problematic result DataFrame.
            code: Source code that produced this result.

        Returns:
            Diagnosis with root cause and suggested fix.
        """
        analysis = self.analyze_dataframe(result_df)

        # First, check for known pitfalls in the code
        lint_results = self.linter.get_critical_issues(code)

        # Look for PANDAS_001 specifically (the most common cause)
        for result in lint_results:
            if result.pitfall.id == "PANDAS_001":
                return Diagnosis(
                    cause=(
                        "Empty DataFrame created by pd.DataFrame(series, columns=[...]). "
                        "When a Series has a MultiIndex, this constructor creates an empty "
                        "DataFrame because it treats the Series as rows, not a column."
                    ),
                    fix=result.suggested_fix,
                    pitfall_id="PANDAS_001",
                    confidence=0.95,
                    details={
                        "line_number": result.line_number,
                        "problematic_code": result.code_snippet,
                        "analysis": analysis.__dict__,
                    },
                )

        # Check for missing $ prefix
        for result in lint_results:
            if result.pitfall.id == "PANDAS_004":
                return Diagnosis(
                    cause=(
                        "KeyError likely caused by missing $ prefix in column access. "
                        "Qlib data uses $close, $open, etc. not close, open."
                    ),
                    fix=result.suggested_fix,
                    pitfall_id="PANDAS_004",
                    confidence=0.9,
                    details={
                        "line_number": result.line_number,
                        "problematic_code": result.code_snippet,
                        "analysis": analysis.__dict__,
                    },
                )

        # Heuristic: Check if code contains pd.DataFrame(..., columns=...)
        if self._contains_dataframe_constructor_with_columns(code):
            return Diagnosis(
                cause=(
                    "Possible issue with pd.DataFrame constructor. When converting a "
                    "Series to DataFrame, use series.to_frame(name='column') instead of "
                    "pd.DataFrame(series, columns=['column'])."
                ),
                fix="Use .to_frame(name='column_name') instead of pd.DataFrame(..., columns=[...])",
                pitfall_id="PANDAS_001",
                confidence=0.7,
                details={"analysis": analysis.__dict__},
            )

        # Generic diagnosis for empty results
        if analysis.is_empty:
            return Diagnosis(
                cause="DataFrame is empty (0 rows). Check data loading and filtering operations.",
                fix="Verify input data exists and filtering conditions don't eliminate all rows.",
                confidence=0.5,
                details={"analysis": analysis.__dict__},
            )

        if analysis.is_all_nan:
            return Diagnosis(
                cause=(
                    f"All {analysis.row_count} rows contain NaN values. "
                    "This may indicate a calculation error or data alignment issue."
                ),
                fix="Check that calculations don't produce NaN (e.g., division by zero, log of negative).",
                confidence=0.6,
                details={"analysis": analysis.__dict__},
            )

        # Unknown cause
        return Diagnosis(
            cause="Unable to determine specific cause of the issue.",
            fix="Review the code for potential data transformation errors.",
            confidence=0.3,
            details={"analysis": analysis.__dict__},
        )

    def diagnose_nan_values(self, result_df: pd.DataFrame, code: str) -> Diagnosis:
        """Diagnose why a DataFrame has excessive NaN values.

        Args:
            result_df: DataFrame with NaN issues.
            code: Source code that produced this result.

        Returns:
            Diagnosis with root cause and suggested fix.
        """
        analysis = self.analyze_dataframe(result_df)

        if analysis.nan_ratio > 0.9:
            # Almost all NaN - likely a fundamental issue
            lint_results = self.linter.lint_code(code)
            if lint_results:
                first_issue = lint_results[0]
                return Diagnosis(
                    cause=f"High NaN ratio ({analysis.nan_ratio:.1%}) may be caused by: {first_issue.pitfall.description}",
                    fix=first_issue.suggested_fix,
                    pitfall_id=first_issue.pitfall.id,
                    confidence=0.7,
                    details={"nan_ratio": analysis.nan_ratio, "analysis": analysis.__dict__},
                )

        if analysis.nan_ratio > 0.5:
            return Diagnosis(
                cause=(
                    f"More than half ({analysis.nan_ratio:.1%}) of values are NaN. "
                    "This could indicate lookback period issues or data gaps."
                ),
                fix="Check if lookback periods (e.g., .shift(20)) exceed available data range.",
                confidence=0.6,
                details={"nan_ratio": analysis.nan_ratio, "analysis": analysis.__dict__},
            )

        return Diagnosis(
            cause=f"Some NaN values present ({analysis.nan_ratio:.1%}).",
            fix="Some NaN is normal at series edges due to rolling/shift operations.",
            confidence=0.8,
            details={"nan_ratio": analysis.nan_ratio, "analysis": analysis.__dict__},
        )

    def _contains_dataframe_constructor_with_columns(self, code: str) -> bool:
        """Check if code contains pd.DataFrame(..., columns=...) pattern."""
        # Pattern: pd.DataFrame followed by columns= anywhere in the call
        pattern = r"pd\.DataFrame\s*\([^)]*columns\s*="
        return bool(re.search(pattern, code))

    def format_diagnosis(self, diagnosis: Diagnosis) -> str:
        """Format a diagnosis as a human-readable string.

        Args:
            diagnosis: Diagnosis to format.

        Returns:
            Formatted string.
        """
        lines = [
            "=== Diagnosis ===",
            f"Cause: {diagnosis.cause}",
            f"Suggested Fix: {diagnosis.fix}",
        ]

        if diagnosis.pitfall_id:
            pitfall = get_pitfall_by_id(diagnosis.pitfall_id)
            if pitfall:
                lines.extend([
                    "",
                    f"Related Pitfall: [{pitfall.id}] {pitfall.name}",
                    f"Bad Example: {pitfall.bad_example}",
                    f"Good Example: {pitfall.good_example}",
                ])

        lines.append(f"Confidence: {diagnosis.confidence:.0%}")

        return "\n".join(lines)


def diagnose_factor_result(result_df: pd.DataFrame, code: str) -> Diagnosis:
    """Convenience function to diagnose factor calculation results.

    Args:
        result_df: Factor result DataFrame.
        code: Factor implementation source code.

    Returns:
        Diagnosis with root cause and fix suggestion.
    """
    diagnostics = RuntimeDiagnostics()

    if result_df.empty:
        return diagnostics.diagnose_empty_dataframe(result_df, code)

    # Check for all-NaN case
    if result_df.isna().all().all():
        return diagnostics.diagnose_nan_values(result_df, code)

    # Check for high NaN ratio
    nan_ratio = result_df.isna().sum().sum() / result_df.size
    if nan_ratio > 0.5:
        return diagnostics.diagnose_nan_values(result_df, code)

    # No obvious issues
    return Diagnosis(
        cause="No obvious issues detected in factor result.",
        fix="Factor result appears normal.",
        confidence=1.0,
    )
