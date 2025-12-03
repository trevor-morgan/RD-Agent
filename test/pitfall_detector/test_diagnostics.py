"""Tests for the runtime diagnostics module."""

import pandas as pd
import pytest
from rdagent.components.coder.pitfall_detector import (
    RuntimeDiagnostics,
    diagnose_factor_result,
)


class TestRuntimeDiagnostics:
    """Test cases for RuntimeDiagnostics."""

    def test_analyze_empty_dataframe(self):
        """Test analysis of empty DataFrame."""
        df = pd.DataFrame(columns=["factor"])
        diagnostics = RuntimeDiagnostics()
        analysis = diagnostics.analyze_dataframe(df)

        assert analysis.is_empty is True
        assert analysis.is_all_nan is True
        assert analysis.row_count == 0
        assert analysis.column_count == 1

    def test_analyze_normal_dataframe(self):
        """Test analysis of normal DataFrame."""
        df = pd.DataFrame({"factor": [1.0, 2.0, 3.0]})
        diagnostics = RuntimeDiagnostics()
        analysis = diagnostics.analyze_dataframe(df)

        assert analysis.is_empty == False
        assert analysis.is_all_nan == False
        assert analysis.row_count == 3
        assert analysis.nan_ratio == 0.0

    def test_analyze_dataframe_with_nan(self):
        """Test analysis of DataFrame with NaN values."""
        df = pd.DataFrame({"factor": [1.0, float("nan"), 3.0]})
        diagnostics = RuntimeDiagnostics()
        analysis = diagnostics.analyze_dataframe(df)

        assert analysis.is_empty == False
        assert analysis.is_all_nan == False
        assert analysis.nan_ratio == pytest.approx(1 / 3)

    def test_analyze_all_nan_dataframe(self):
        """Test analysis of all-NaN DataFrame."""
        df = pd.DataFrame({"factor": [float("nan"), float("nan")]})
        diagnostics = RuntimeDiagnostics()
        analysis = diagnostics.analyze_dataframe(df)

        assert analysis.is_all_nan == True
        assert analysis.nan_ratio == 1.0

    def test_diagnose_empty_with_dataframe_constructor(self):
        """Test diagnosis of empty DataFrame caused by constructor pitfall."""
        empty_df = pd.DataFrame(columns=["Momentum"])
        code = """
import pandas as pd
momentum = df["$close"] / df["$close"].shift(20) - 1
result = pd.DataFrame(momentum, columns=["Momentum"])
"""
        diagnostics = RuntimeDiagnostics()
        diagnosis = diagnostics.diagnose_empty_dataframe(empty_df, code)

        assert diagnosis.pitfall_id == "PANDAS_001"
        assert diagnosis.confidence >= 0.9
        assert "to_frame" in diagnosis.fix.lower()

    def test_diagnose_empty_with_missing_dollar(self):
        """Test diagnosis when missing $ prefix in column access."""
        empty_df = pd.DataFrame(columns=["Momentum"])
        code = """
import pandas as pd
momentum = df["close"] / df["close"].shift(20) - 1
result = momentum.to_frame(name="Momentum")
"""
        diagnostics = RuntimeDiagnostics()
        diagnosis = diagnostics.diagnose_empty_dataframe(empty_df, code)

        assert diagnosis.pitfall_id == "PANDAS_004"
        assert "$" in diagnosis.fix

    def test_diagnose_nan_values(self):
        """Test diagnosis of excessive NaN values."""
        df = pd.DataFrame({"factor": [float("nan")] * 10})
        code = "result = some_calculation()"

        diagnostics = RuntimeDiagnostics()
        diagnosis = diagnostics.diagnose_nan_values(df, code)

        assert "nan" in diagnosis.cause.lower()
        assert diagnosis.confidence >= 0.6

    def test_format_diagnosis(self):
        """Test formatting of diagnosis output."""
        empty_df = pd.DataFrame()
        code = """result = pd.DataFrame(x, columns=["Y"])"""

        diagnostics = RuntimeDiagnostics()
        diagnosis = diagnostics.diagnose_empty_dataframe(empty_df, code)
        formatted = diagnostics.format_diagnosis(diagnosis)

        assert "Cause:" in formatted
        assert "Suggested Fix:" in formatted
        assert "PANDAS_001" in formatted


class TestDiagnoseFactorResult:
    """Test the diagnose_factor_result convenience function."""

    def test_diagnose_empty_result(self):
        """Test diagnosis of empty factor result."""
        empty_df = pd.DataFrame()
        code = """result = pd.DataFrame(x, columns=["Y"])"""

        diagnosis = diagnose_factor_result(empty_df, code)
        assert diagnosis.pitfall_id is not None

    def test_diagnose_normal_result(self):
        """Test diagnosis of normal factor result."""
        df = pd.DataFrame({"factor": [1.0, 2.0, 3.0]})
        code = """result = factor.to_frame(name="factor")"""

        diagnosis = diagnose_factor_result(df, code)
        assert diagnosis.confidence == 1.0
        assert "normal" in diagnosis.cause.lower() or "no obvious" in diagnosis.cause.lower()
