"""Pitfall pattern definitions for pandas anti-patterns detection.

This module defines common pandas coding pitfalls that can cause silent failures,
particularly in factor coding where MultiIndex Series are common.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class PitfallPattern:
    """Definition of a pandas coding pitfall pattern.

    Attributes:
        id: Unique identifier for the pitfall (e.g., "PANDAS_001")
        name: Short descriptive name
        description: Detailed explanation of the pitfall
        detection_ast_pattern: Pattern description for AST-based detection
        detection_runtime: Optional function to detect at runtime
        correction_template: Template showing how to fix the issue
        bad_example: Example of incorrect code
        good_example: Example of correct code
        severity: How critical this pitfall is
    """

    id: str
    name: str
    description: str
    detection_ast_pattern: str
    correction_template: str
    bad_example: str
    good_example: str
    severity: Literal["critical", "warning", "info"] = "warning"
    detection_runtime: Callable[[pd.DataFrame, str], bool] | None = None


@dataclass
class LintResult:
    """Result from linting code for pitfall patterns.

    Attributes:
        pitfall: The detected pitfall pattern
        line_number: Line number where the pitfall was detected
        column: Column offset in the line
        code_snippet: The problematic code snippet
        suggested_fix: Suggested code to fix the issue
    """

    pitfall: PitfallPattern
    line_number: int
    column: int
    code_snippet: str
    suggested_fix: str

    @property
    def severity(self) -> Literal["critical", "warning", "info"]:
        return self.pitfall.severity

    def __str__(self) -> str:
        return (
            f"[{self.pitfall.id}] {self.pitfall.name} at line {self.line_number}:{self.column}\n"
            f"  Problem: {self.code_snippet}\n"
            f"  Fix: {self.suggested_fix}"
        )


@dataclass
class Diagnosis:
    """Runtime diagnosis of a DataFrame issue.

    Attributes:
        cause: Root cause description
        pitfall_id: ID of the detected pitfall pattern (if matched)
        fix: Suggested fix
        confidence: How confident we are in this diagnosis (0.0-1.0)
        details: Additional diagnostic details
    """

    cause: str
    fix: str
    pitfall_id: str | None = None
    confidence: float = 1.0
    details: dict = field(default_factory=dict)


# Common pandas pitfall patterns
PANDAS_PITFALLS: list[PitfallPattern] = [
    PitfallPattern(
        id="PANDAS_001",
        name="multiindex_series_to_dataframe",
        description=(
            "Using pd.DataFrame(series, columns=[...]) with a MultiIndex Series creates "
            "an empty DataFrame because the constructor treats the Series as data rows, "
            "not as a column. The MultiIndex becomes misaligned with the new column names."
        ),
        detection_ast_pattern="pd.DataFrame(<series_var>, columns=[...])",
        correction_template="Use series.to_frame(name='column_name') instead of pd.DataFrame(series, columns=[...])",
        bad_example='result = pd.DataFrame(factor_series, columns=["FactorName"])',
        good_example='result = factor_series.to_frame(name="FactorName")',
        severity="critical",
    ),
    PitfallPattern(
        id="PANDAS_002",
        name="inplace_on_view",
        description=(
            "Using inplace=True on a DataFrame view (slice) may not modify the original "
            "DataFrame and can raise SettingWithCopyWarning. This often silently fails."
        ),
        detection_ast_pattern="df[...].method(..., inplace=True)",
        correction_template="Avoid inplace=True on slices. Use df.loc[...] = ... or reassign the result.",
        bad_example='df[df["col"] > 0].fillna(0, inplace=True)',
        good_example='df.loc[df["col"] > 0, "col"] = df.loc[df["col"] > 0, "col"].fillna(0)',
        severity="warning",
    ),
    PitfallPattern(
        id="PANDAS_003",
        name="chained_assignment",
        description=(
            "Chained indexing like df['a']['b'] = value may not work as expected. "
            "Use df.loc[row, col] = value instead for reliable assignment."
        ),
        detection_ast_pattern="df[...][...] = ...",
        correction_template="Use df.loc[row_indexer, col_indexer] = value for assignments.",
        bad_example='df["a"]["b"] = 5',
        good_example='df.loc[:, "a"].loc[:, "b"] = 5  # or better: df.loc[:, ("a", "b")] = 5',
        severity="warning",
    ),
    PitfallPattern(
        id="PANDAS_004",
        name="missing_dollar_prefix",
        description=(
            "Qlib data uses $ prefix for column names ($open, $close, $high, $low, $volume, $factor). "
            "Accessing columns without $ prefix will raise KeyError or return wrong data."
        ),
        detection_ast_pattern='df["close"] or df["open"] (without $ prefix)',
        correction_template='Use df["$close"], df["$open"], etc. for Qlib data columns.',
        bad_example='momentum = df["close"] / df["close"].shift(20) - 1',
        good_example='momentum = df["$close"] / df["$close"].shift(20) - 1',
        severity="critical",
    ),
    PitfallPattern(
        id="PANDAS_005",
        name="groupby_transform_vs_apply",
        description=(
            "Using groupby().apply() when groupby().transform() is needed can cause "
            "index alignment issues. transform() preserves the original index."
        ),
        detection_ast_pattern="df.groupby(...).apply(...) when result should align with original index",
        correction_template="Use groupby().transform() to preserve index alignment.",
        bad_example='df["normalized"] = df.groupby("instrument").apply(lambda x: (x - x.mean()) / x.std())',
        good_example='df["normalized"] = df.groupby("instrument")["value"].transform(lambda x: (x - x.mean()) / x.std())',
        severity="warning",
    ),
]


def get_pitfall_by_id(pitfall_id: str) -> PitfallPattern | None:
    """Get a pitfall pattern by its ID."""
    for pitfall in PANDAS_PITFALLS:
        if pitfall.id == pitfall_id:
            return pitfall
    return None


def get_critical_pitfalls() -> list[PitfallPattern]:
    """Get all critical severity pitfall patterns."""
    return [p for p in PANDAS_PITFALLS if p.severity == "critical"]
