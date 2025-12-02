"""Backtest service supporting Qlib and VectorBT engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from loguru import logger

from rdagent_lab.analytics.metrics import PerformanceAnalyzer, generate_tearsheet
from rdagent_lab.core.exceptions import ConfigurationError


@dataclass
class BacktestConfig:
    strategy_type: str = "topk"
    topk: int = 30
    threshold: float = 0.0
    benchmark: str = "SPY"
    account: float = 1_000_000
    exchange_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    )
    start_time: str = "2017-01-01"
    end_time: str = "2020-08-01"
    report_output: str | None = None
    engine: Literal["qlib", "vectorbt"] = "qlib"


@dataclass
class BacktestResult:
    returns: pd.Series
    portfolio_value: pd.Series
    positions: pd.DataFrame | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    analyzer: PerformanceAnalyzer | None = None
    benchmark_returns: pd.Series | None = None
    config: BacktestConfig | None = None
    report_path: str | None = None


class BacktestService:
    """High-level service for running backtests."""

    def run(
        self,
        predictions: pd.Series,
        prices: pd.DataFrame | None = None,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        config = config or BacktestConfig()
        if config.engine == "qlib":
            return self.run_qlib(predictions, config)
        if prices is None:
            raise ConfigurationError("prices", "Price data required for VectorBT engine")
        return self.run_vectorbt(predictions, prices, config)

    def run_qlib(self, predictions: pd.Series, config: BacktestConfig) -> BacktestResult:
        logger.info(f"Running Qlib backtest: strategy={config.strategy_type}")
        try:
            from qlib.contrib.evaluate import backtest as qlib_backtest
        except ImportError as exc:
            raise ConfigurationError(
                "qlib", f"Qlib not available. Install with: pip install rdagent[quant-lab]. Error: {exc}"
            ) from exc

        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": predictions, "topk": config.topk, "n_drop": config.topk // 10},
        }
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        }
        portfolio_metric, indicator_dict = qlib_backtest(
            pred=predictions,
            strategy=strategy_config,
            executor=executor_config,
            account=config.account,
            benchmark=config.benchmark,
            exchange_kwargs=config.exchange_kwargs,
        )
        returns = portfolio_metric["return"].dropna()
        portfolio_value = (1 + returns).cumprod() * config.account
        benchmark_returns = portfolio_metric["bench"].dropna() if "bench" in portfolio_metric.columns else None
        analyzer = PerformanceAnalyzer(returns, benchmark=benchmark_returns)
        metrics = analyzer.summary()
        report_path = None
        if config.report_output:
            analyzer.generate_report(config.report_output)
            report_path = config.report_output
            logger.info(f"Report saved to: {report_path}")
        logger.info(f"Backtest complete. Sharpe: {metrics.get('sharpe', 0):.2f}")
        return BacktestResult(
            returns=returns,
            portfolio_value=portfolio_value,
            metrics=metrics,
            analyzer=analyzer,
            benchmark_returns=benchmark_returns,
            config=config,
            report_path=report_path,
        )

    def run_vectorbt(
        self,
        signals: pd.Series | pd.DataFrame,
        prices: pd.DataFrame,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        config = config or BacktestConfig(engine="vectorbt")
        logger.info("Running VectorBT backtest")
        try:
            import vectorbt as vbt
        except ImportError as exc:
            raise ConfigurationError(
                "vectorbt", f"VectorBT not available. Install with: pip install rdagent[backtest]. Error: {exc}"
            ) from exc

        if isinstance(signals, pd.Series) and signals.index.nlevels > 1:
            signals = signals.unstack()
        entries = signals > config.threshold
        exits = signals < -config.threshold
        common_cols = prices.columns.intersection(signals.columns) if hasattr(signals, "columns") else []
        if len(common_cols) == 0:
            close_prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
        else:
            close_prices = prices[common_cols]
        portfolio = vbt.Portfolio.from_signals(
            close_prices,
            entries=entries,
            exits=exits,
            init_cash=config.account,
            fees=config.exchange_kwargs.get("open_cost", 0.0005),
            slippage=0.001,
            freq="D",
        )
        returns = portfolio.returns()
        portfolio_value = portfolio.value()
        try:
            stats = portfolio.stats()
            stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception:  # noqa: BLE001
            stats_dict = {}

        if hasattr(returns, "columns") and len(returns.columns) > 1:
            returns = returns.mean(axis=1)
        if hasattr(portfolio_value, "columns") and len(portfolio_value.columns) > 1:
            portfolio_value = portfolio_value.mean(axis=1)

        analyzer = PerformanceAnalyzer(returns)
        metrics = analyzer.summary()

        def safe_get(d: dict, key: str, default: float = 0) -> float:
            val = d.get(key, default)
            if hasattr(val, "mean"):
                return float(val.mean())
            return float(val) if val is not None else default

        metrics.update(
            {
                "total_trades": int(safe_get(stats_dict, "Total Trades", 0)),
                "win_rate_vbt": safe_get(stats_dict, "Win Rate [%]", 0) / 100,
                "avg_trade_return": safe_get(stats_dict, "Avg Trade [%]", 0) / 100,
            }
        )
        report_path = None
        if config.report_output:
            generate_tearsheet(returns, output=config.report_output)
            report_path = config.report_output

        logger.info(f"VectorBT backtest complete. Sharpe: {metrics.get('sharpe', 0):.2f}")
        return BacktestResult(
            returns=returns,
            portfolio_value=portfolio_value,
            metrics=metrics,
            analyzer=analyzer,
            config=config,
            report_path=report_path,
        )
