"""Performance metrics and analytics wrapping QuantStats and empyrical."""

from __future__ import annotations

from pathlib import Path

import empyrical as ep
import numpy as np
import pandas as pd
import quantstats as qs


def calculate_sharpe(returns: pd.Series, risk_free: float = 0.0, periods: int = 252) -> float:
    return ep.sharpe_ratio(returns, risk_free=risk_free, period="daily", annualization=periods)


def calculate_sortino(returns: pd.Series, required_return: float = 0.0, periods: int = 252) -> float:
    return ep.sortino_ratio(returns, required_return=required_return, period="daily", annualization=periods)


def calculate_max_drawdown(returns: pd.Series) -> float:
    return ep.max_drawdown(returns)


def calculate_calmar(returns: pd.Series, periods: int = 252) -> float:
    return ep.calmar_ratio(returns, period="daily", annualization=periods)


def calculate_omega(returns: pd.Series, required_return: float = 0.0, periods: int = 252) -> float:
    return ep.omega_ratio(returns, required_return=required_return, annualization=periods)


def calculate_cagr(returns: pd.Series, periods: int = 252) -> float:
    return ep.cagr(returns, period="daily", annualization=periods)


def calculate_volatility(returns: pd.Series, periods: int = 252) -> float:
    return ep.annual_volatility(returns, period="daily", annualization=periods)


def generate_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output: str | Path | None = None,
    title: str = "Strategy Performance",
) -> None:
    if output:
        qs.reports.html(returns, benchmark=benchmark, output=str(output), title=title)
    else:
        qs.reports.html(returns, benchmark=benchmark, title=title)


class PerformanceAnalyzer:
    """Convenience wrapper around quantstats/empyrical for common metrics."""

    def __init__(
        self,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        risk_free: float = 0.0,
        periods: int = 252,
    ) -> None:
        self.returns = returns.dropna()
        self.benchmark = benchmark.dropna() if benchmark is not None else None
        self.risk_free = risk_free
        self.periods = periods

    @property
    def sharpe(self) -> float:
        return calculate_sharpe(self.returns, self.risk_free, self.periods)

    @property
    def sortino(self) -> float:
        return calculate_sortino(self.returns, periods=self.periods)

    @property
    def max_drawdown(self) -> float:
        return calculate_max_drawdown(self.returns)

    @property
    def calmar(self) -> float:
        return calculate_calmar(self.returns, self.periods)

    @property
    def omega(self) -> float:
        return calculate_omega(self.returns, periods=self.periods)

    @property
    def cagr(self) -> float:
        return calculate_cagr(self.returns, self.periods)

    @property
    def volatility(self) -> float:
        return calculate_volatility(self.returns, self.periods)

    @property
    def total_return(self) -> float:
        return ep.cum_returns_final(self.returns)

    @property
    def win_rate(self) -> float:
        return (self.returns > 0).mean()

    @property
    def profit_factor(self) -> float:
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        if losses == 0:
            return np.inf
        return gains / losses

    def alpha_beta(self) -> tuple[float, float]:
        if self.benchmark is None:
            raise ValueError("Benchmark required for alpha/beta calculation")
        alpha, beta = ep.alpha_beta(
            self.returns,
            self.benchmark,
            risk_free=self.risk_free,
            period="daily",
            annualization=self.periods,
        )
        return alpha, beta

    def summary(self) -> dict[str, float]:
        stats = {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
        }
        if self.benchmark is not None:
            try:
                alpha, beta = self.alpha_beta()
                stats["alpha"] = alpha
                stats["beta"] = beta
            except Exception:
                pass
        return stats

    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary()])

    def generate_report(self, output: str | Path | None = None, title: str = "Strategy Performance") -> None:
        generate_tearsheet(self.returns, benchmark=self.benchmark, output=output, title=title)
