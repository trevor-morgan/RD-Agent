"""Tests for the pitfall linter module."""


from rdagent.components.coder.pitfall_detector import (
    PANDAS_PITFALLS,
    PitfallLinter,
    has_critical_pitfalls,
    lint_factor_code,
)


class TestPitfallLinter:
    """Test cases for PitfallLinter."""

    def test_detect_dataframe_constructor_with_columns(self):
        """Test detection of pd.DataFrame(series, columns=[...]) pattern."""
        code = """
import pandas as pd
momentum = df["$close"] / df["$close"].shift(20) - 1
result = pd.DataFrame(momentum, columns=["Momentum"])
"""
        linter = PitfallLinter()
        results = linter.lint_code(code)

        pandas_001_results = [r for r in results if r.pitfall.id == "PANDAS_001"]
        assert len(pandas_001_results) == 1
        assert pandas_001_results[0].line_number == 4
        assert "to_frame" in pandas_001_results[0].suggested_fix.lower()

    def test_detect_missing_dollar_prefix(self):
        """Test detection of missing $ prefix in Qlib column access."""
        code = """
import pandas as pd
# Missing $ prefix
close_price = df["close"]
open_price = df["open"]
"""
        linter = PitfallLinter()
        results = linter.lint_code(code)

        pandas_004_results = [r for r in results if r.pitfall.id == "PANDAS_004"]
        assert len(pandas_004_results) == 2
        assert all("$" in r.suggested_fix for r in pandas_004_results)

    def test_no_false_positive_with_dollar_prefix(self):
        """Test that correct code with $ prefix doesn't trigger warnings."""
        code = """
import pandas as pd
# Correct usage
close_price = df["$close"]
open_price = df["$open"]
"""
        linter = PitfallLinter()
        results = linter.lint_code(code)

        pandas_004_results = [r for r in results if r.pitfall.id == "PANDAS_004"]
        assert len(pandas_004_results) == 0

    def test_no_false_positive_with_to_frame(self):
        """Test that correct code with to_frame() doesn't trigger PANDAS_001."""
        code = """
import pandas as pd
momentum = df["$close"] / df["$close"].shift(20) - 1
result = momentum.to_frame(name="Momentum")
"""
        linter = PitfallLinter()
        results = linter.lint_code(code)

        pandas_001_results = [r for r in results if r.pitfall.id == "PANDAS_001"]
        assert len(pandas_001_results) == 0

    def test_detect_inplace_on_slice(self):
        """Test detection of inplace=True on DataFrame slice."""
        code = """
import pandas as pd
df[df["col"] > 0].fillna(0, inplace=True)
"""
        linter = PitfallLinter()
        results = linter.lint_code(code)

        pandas_002_results = [r for r in results if r.pitfall.id == "PANDAS_002"]
        assert len(pandas_002_results) == 1

    def test_syntax_error_handling(self):
        """Test that invalid Python code doesn't crash the linter."""
        code = "def foo(:\n  pass"  # Syntax error
        linter = PitfallLinter()
        results = linter.lint_code(code)

        assert results == []  # Should return empty list, not crash

    def test_get_critical_issues(self):
        """Test filtering for critical issues only."""
        code = """
import pandas as pd
# Critical: missing $ prefix
close = df["close"]
# Critical: DataFrame constructor
result = pd.DataFrame(factor, columns=["Factor"])
# Warning: inplace on slice
df[mask].dropna(inplace=True)
"""
        linter = PitfallLinter()
        critical = linter.get_critical_issues(code)

        # PANDAS_001 and PANDAS_004 are critical, PANDAS_002 is warning
        critical_ids = {r.pitfall.id for r in critical}
        assert "PANDAS_001" in critical_ids
        assert "PANDAS_004" in critical_ids
        # PANDAS_002 is warning severity, should not be in critical
        assert all(r.severity == "critical" for r in critical)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_lint_factor_code(self):
        """Test the lint_factor_code convenience function."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""
        results = lint_factor_code(code)

        assert len(results) >= 1
        assert any(r.pitfall.id == "PANDAS_001" for r in results)

    def test_has_critical_pitfalls_true(self):
        """Test has_critical_pitfalls returns True for problematic code."""
        code = """result = pd.DataFrame(x, columns=["Y"])"""
        assert has_critical_pitfalls(code) is True

    def test_has_critical_pitfalls_false(self):
        """Test has_critical_pitfalls returns False for clean code."""
        code = """result = x.to_frame(name="Y")"""
        assert has_critical_pitfalls(code) is False


class TestPitfallPatterns:
    """Test pitfall pattern definitions."""

    def test_all_patterns_have_required_fields(self):
        """Test that all patterns have required fields."""
        for pattern in PANDAS_PITFALLS:
            assert pattern.id, "Pattern must have an id"
            assert pattern.name, "Pattern must have a name"
            assert pattern.description, "Pattern must have a description"
            assert pattern.correction_template, "Pattern must have a correction_template"
            assert pattern.bad_example, "Pattern must have a bad_example"
            assert pattern.good_example, "Pattern must have a good_example"
            assert pattern.severity in ("critical", "warning", "info")

    def test_pattern_ids_are_unique(self):
        """Test that all pattern IDs are unique."""
        ids = [p.id for p in PANDAS_PITFALLS]
        assert len(ids) == len(set(ids)), "Pattern IDs must be unique"

    def test_critical_patterns_exist(self):
        """Test that critical patterns exist."""
        critical = [p for p in PANDAS_PITFALLS if p.severity == "critical"]
        assert len(critical) >= 2, "Should have at least 2 critical patterns"
