"""Tests for BacktestService."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from rdagent_lab.core.exceptions import ConfigurationError
from rdagent_lab.services.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestService,
)


# --- BacktestConfig tests ---


def test_backtest_config_default_values() -> None:
    """Should have sensible defaults."""
    config = BacktestConfig()
    assert config.strategy_type == "topk"
    assert config.topk == 30
    assert config.threshold == 0.0
    assert config.engine == "qlib"
    assert config.account == 1_000_000


def test_backtest_config_vectorbt_engine() -> None:
    """Should accept vectorbt engine."""
    config = BacktestConfig(engine="vectorbt")
    assert config.engine == "vectorbt"


def test_backtest_config_custom_topk() -> None:
    """Should accept custom topk value."""
    config = BacktestConfig(topk=50)
    assert config.topk == 50


def test_backtest_config_exchange_kwargs_defaults() -> None:
    """Should have default exchange kwargs."""
    config = BacktestConfig()
    assert "limit_threshold" in config.exchange_kwargs
    assert "deal_price" in config.exchange_kwargs
    assert config.exchange_kwargs["open_cost"] == 0.0005


def test_backtest_config_custom_exchange_kwargs() -> None:
    """Should accept custom exchange kwargs."""
    config = BacktestConfig(exchange_kwargs={"open_cost": 0.001, "close_cost": 0.002})
    assert config.exchange_kwargs["open_cost"] == 0.001
    assert config.exchange_kwargs["close_cost"] == 0.002


# --- BacktestResult tests ---


def test_backtest_result_required_fields() -> None:
    """Should require returns and portfolio_value."""
    returns = pd.Series([0.01, -0.02, 0.015])
    portfolio_value = pd.Series([100000, 98000, 99470])
    result = BacktestResult(returns=returns, portfolio_value=portfolio_value)
    assert len(result.returns) == 3
    assert len(result.portfolio_value) == 3


def test_backtest_result_optional_fields_default() -> None:
    """Should default optional fields appropriately."""
    returns = pd.Series([0.01])
    portfolio_value = pd.Series([100000])
    result = BacktestResult(returns=returns, portfolio_value=portfolio_value)
    assert result.positions is None
    assert result.metrics == {}
    assert result.analyzer is None
    assert result.benchmark_returns is None
    assert result.config is None
    assert result.report_path is None


def test_backtest_result_with_metrics() -> None:
    """Should accept metrics dict."""
    returns = pd.Series([0.01])
    portfolio_value = pd.Series([100000])
    result = BacktestResult(
        returns=returns,
        portfolio_value=portfolio_value,
        metrics={"sharpe": 1.5, "max_drawdown": -0.15},
    )
    assert result.metrics["sharpe"] == 1.5
    assert result.metrics["max_drawdown"] == -0.15


# --- BacktestService tests ---


def test_backtest_service_run_requires_prices_for_vectorbt(sample_predictions: pd.Series) -> None:
    """Should raise if prices missing for vectorbt."""
    service = BacktestService()
    config = BacktestConfig(engine="vectorbt")

    with pytest.raises(ConfigurationError, match="Price data required"):
        service.run(sample_predictions, prices=None, config=config)


def test_backtest_service_run_qlib_engine_default(sample_predictions: pd.Series) -> None:
    """Should default to qlib engine."""
    service = BacktestService()
    config = BacktestConfig()

    with patch.object(service, "run_qlib") as mock_run_qlib:
        mock_run_qlib.return_value = BacktestResult(
            returns=pd.Series([0.01]),
            portfolio_value=pd.Series([100000]),
        )
        service.run(sample_predictions, config=config)
        mock_run_qlib.assert_called_once()


def test_backtest_service_run_vectorbt_engine_with_prices(
    sample_predictions: pd.Series, sample_prices: pd.DataFrame
) -> None:
    """Should use vectorbt engine when specified with prices."""
    service = BacktestService()
    config = BacktestConfig(engine="vectorbt")

    with patch.object(service, "run_vectorbt") as mock_run_vectorbt:
        mock_run_vectorbt.return_value = BacktestResult(
            returns=pd.Series([0.01]),
            portfolio_value=pd.Series([100000]),
        )
        service.run(sample_predictions, prices=sample_prices, config=config)
        mock_run_vectorbt.assert_called_once()


def test_backtest_service_run_default_config_when_none(sample_predictions: pd.Series) -> None:
    """Should create default config when None provided."""
    service = BacktestService()

    with patch.object(service, "run_qlib") as mock_run_qlib:
        mock_run_qlib.return_value = BacktestResult(
            returns=pd.Series([0.01]),
            portfolio_value=pd.Series([100000]),
        )
        service.run(sample_predictions, config=None)
        # Should have been called with a BacktestConfig
        call_args = mock_run_qlib.call_args
        assert call_args is not None


def test_backtest_service_run_vectorbt_with_mock(
    sample_predictions: pd.Series, sample_prices: pd.DataFrame
) -> None:
    """Should handle vectorbt backtest with mocked portfolio."""
    import sys

    # Create mock vectorbt module
    mock_vbt = Mock()
    mock_portfolio = Mock()
    mock_portfolio.returns.return_value = pd.Series([0.01] * 10)
    mock_portfolio.value.return_value = pd.Series([100000] * 10)
    mock_portfolio.stats.return_value = {"Total Trades": 10, "Win Rate [%]": 55}
    mock_vbt.Portfolio.from_signals.return_value = mock_portfolio

    # Patch sys.modules to provide fake vectorbt
    with patch.dict(sys.modules, {"vectorbt": mock_vbt}):
        service = BacktestService()
        config = BacktestConfig(engine="vectorbt", threshold=0.0)

        result = service.run_vectorbt(sample_predictions, sample_prices, config)

        assert result.returns is not None
        assert result.portfolio_value is not None
        assert len(result.metrics) > 0
