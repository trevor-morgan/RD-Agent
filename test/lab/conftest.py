"""Shared fixtures for lab tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns() -> pd.Series:
    """Generate sample return series for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.randn(252) * 0.02, index=dates)
    return returns


@pytest.fixture
def sample_benchmark() -> pd.Series:
    """Generate sample benchmark return series for testing."""
    np.random.seed(123)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.randn(252) * 0.015, index=dates)
    return returns


@pytest.fixture
def sample_predictions() -> pd.Series:
    """Generate sample prediction series with multi-index for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    index = pd.MultiIndex.from_product([dates, symbols], names=["datetime", "instrument"])
    predictions = pd.Series(np.random.randn(len(index)) * 0.1, index=index)
    return predictions


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate sample price dataframe for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    # Generate random walk prices
    prices = pd.DataFrame(
        np.abs(np.random.randn(100, 3).cumsum(axis=0)) + 100,
        index=dates,
        columns=symbols,
    )
    return prices


@pytest.fixture
def positive_returns() -> pd.Series:
    """Generate mostly positive returns for testing edge cases."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(np.abs(np.random.randn(100)) * 0.01, index=dates)
    return returns


@pytest.fixture
def negative_returns() -> pd.Series:
    """Generate mostly negative returns for testing edge cases."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.Series(-np.abs(np.random.randn(100)) * 0.01, index=dates)
    return returns
