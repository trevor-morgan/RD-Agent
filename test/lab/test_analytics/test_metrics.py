"""Tests for analytics metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdagent_lab.analytics.metrics import (
    PerformanceAnalyzer,
    _HAS_EMPYRICAL,
    _HAS_QUANTSTATS,
    calculate_max_drawdown,
    calculate_sharpe,
)


# Skip all tests if empyrical not installed
pytestmark = pytest.mark.skipif(not _HAS_EMPYRICAL, reason="empyrical not installed")


# --- Standalone metric function tests ---


def test_calculate_sharpe_returns_float(sample_returns: pd.Series) -> None:
    """Should calculate Sharpe ratio as float."""
    sharpe = calculate_sharpe(sample_returns)
    assert isinstance(sharpe, (float, np.floating))


def test_calculate_sharpe_with_risk_free(sample_returns: pd.Series) -> None:
    """Should accept risk_free parameter."""
    sharpe_zero = calculate_sharpe(sample_returns, risk_free=0.0)
    sharpe_positive = calculate_sharpe(sample_returns, risk_free=0.02)
    # With positive risk-free rate, Sharpe should be lower
    assert sharpe_positive <= sharpe_zero


def test_calculate_sharpe_with_periods(sample_returns: pd.Series) -> None:
    """Should accept periods parameter for annualization."""
    sharpe_252 = calculate_sharpe(sample_returns, periods=252)
    sharpe_365 = calculate_sharpe(sample_returns, periods=365)
    # Different annualization should give different results
    assert sharpe_252 != sharpe_365


def test_calculate_max_drawdown_returns_float(sample_returns: pd.Series) -> None:
    """Should calculate max drawdown as float."""
    mdd = calculate_max_drawdown(sample_returns)
    assert isinstance(mdd, (float, np.floating))


def test_calculate_max_drawdown_is_negative_or_zero(sample_returns: pd.Series) -> None:
    """Max drawdown should be negative or zero."""
    mdd = calculate_max_drawdown(sample_returns)
    assert mdd <= 0


def test_calculate_max_drawdown_positive_returns(positive_returns: pd.Series) -> None:
    """Positive-only returns should have zero or minimal drawdown."""
    mdd = calculate_max_drawdown(positive_returns)
    # With all positive returns, drawdown should be 0 or very small
    assert mdd >= -0.01  # Allow small floating point differences


# --- PerformanceAnalyzer tests ---


def test_performance_analyzer_init(sample_returns: pd.Series) -> None:
    """Should initialize with returns."""
    analyzer = PerformanceAnalyzer(sample_returns)
    assert len(analyzer.returns) > 0
    assert analyzer.benchmark is None
    assert analyzer.risk_free == 0.0
    assert analyzer.periods == 252


def test_performance_analyzer_init_with_benchmark(
    sample_returns: pd.Series, sample_benchmark: pd.Series
) -> None:
    """Should accept benchmark returns."""
    analyzer = PerformanceAnalyzer(sample_returns, benchmark=sample_benchmark)
    assert analyzer.benchmark is not None
    assert len(analyzer.benchmark) > 0


def test_performance_analyzer_init_drops_na(sample_returns: pd.Series) -> None:
    """Should drop NaN values from returns."""
    returns_with_na = sample_returns.copy()
    returns_with_na.iloc[0] = np.nan
    returns_with_na.iloc[-1] = np.nan

    analyzer = PerformanceAnalyzer(returns_with_na)
    assert not analyzer.returns.isna().any()
    assert len(analyzer.returns) == len(sample_returns) - 2


def test_performance_analyzer_sharpe_property(sample_returns: pd.Series) -> None:
    """Should calculate Sharpe ratio via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    sharpe = analyzer.sharpe
    assert isinstance(sharpe, (float, np.floating))


def test_performance_analyzer_sortino_property(sample_returns: pd.Series) -> None:
    """Should calculate Sortino ratio via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    sortino = analyzer.sortino
    assert isinstance(sortino, (float, np.floating))


def test_performance_analyzer_max_drawdown_property(sample_returns: pd.Series) -> None:
    """Should calculate max drawdown via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    mdd = analyzer.max_drawdown
    assert isinstance(mdd, (float, np.floating))
    assert mdd <= 0


def test_performance_analyzer_calmar_property(sample_returns: pd.Series) -> None:
    """Should calculate Calmar ratio via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    calmar = analyzer.calmar
    assert isinstance(calmar, (float, np.floating))


def test_performance_analyzer_volatility_property(sample_returns: pd.Series) -> None:
    """Should calculate volatility via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    vol = analyzer.volatility
    assert isinstance(vol, (float, np.floating))
    assert vol >= 0  # Volatility should be non-negative


def test_performance_analyzer_total_return_property(sample_returns: pd.Series) -> None:
    """Should calculate total return via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    total_ret = analyzer.total_return
    assert isinstance(total_ret, (float, np.floating))


def test_performance_analyzer_win_rate_property(sample_returns: pd.Series) -> None:
    """Should calculate win rate via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    win_rate = analyzer.win_rate
    assert isinstance(win_rate, float)
    assert 0 <= win_rate <= 1


def test_performance_analyzer_profit_factor_property(sample_returns: pd.Series) -> None:
    """Should calculate profit factor via property."""
    analyzer = PerformanceAnalyzer(sample_returns)
    pf = analyzer.profit_factor
    assert isinstance(pf, (float, np.floating))
    assert pf >= 0  # Profit factor should be non-negative


def test_performance_analyzer_profit_factor_no_losses(positive_returns: pd.Series) -> None:
    """Profit factor should be inf when no losses."""
    analyzer = PerformanceAnalyzer(positive_returns)
    pf = analyzer.profit_factor
    assert pf == np.inf


def test_performance_analyzer_alpha_beta_requires_benchmark(sample_returns: pd.Series) -> None:
    """Should raise if no benchmark for alpha/beta."""
    analyzer = PerformanceAnalyzer(sample_returns, benchmark=None)

    with pytest.raises(ValueError, match="Benchmark required"):
        analyzer.alpha_beta()


def test_performance_analyzer_alpha_beta_with_benchmark(
    sample_returns: pd.Series, sample_benchmark: pd.Series
) -> None:
    """Should calculate alpha and beta with benchmark."""
    analyzer = PerformanceAnalyzer(sample_returns, benchmark=sample_benchmark)
    alpha, beta = analyzer.alpha_beta()

    assert isinstance(alpha, (float, np.floating))
    assert isinstance(beta, (float, np.floating))


def test_performance_analyzer_summary_returns_dict(sample_returns: pd.Series) -> None:
    """Should return summary dict with all metrics."""
    analyzer = PerformanceAnalyzer(sample_returns)
    summary = analyzer.summary()

    expected_keys = [
        "total_return",
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "volatility",
        "win_rate",
        "profit_factor",
    ]
    for key in expected_keys:
        assert key in summary


def test_performance_analyzer_summary_with_benchmark(
    sample_returns: pd.Series, sample_benchmark: pd.Series
) -> None:
    """Should include alpha/beta in summary when benchmark provided."""
    analyzer = PerformanceAnalyzer(sample_returns, benchmark=sample_benchmark)
    summary = analyzer.summary()

    assert "alpha" in summary
    assert "beta" in summary


def test_performance_analyzer_summary_df(sample_returns: pd.Series) -> None:
    """Should return summary as DataFrame."""
    analyzer = PerformanceAnalyzer(sample_returns)
    df = analyzer.summary_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "sharpe" in df.columns


# --- Import guard tests ---


def test_has_empyrical_flag() -> None:
    """Should have empyrical availability flag."""
    # This test runs so empyrical must be available
    assert _HAS_EMPYRICAL is True


def test_has_quantstats_flag() -> None:
    """Should have quantstats availability flag."""
    # Check if the flag exists (value depends on installation)
    assert isinstance(_HAS_QUANTSTATS, bool)
