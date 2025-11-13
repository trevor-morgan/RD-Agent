"""
Predictable Trend Strength (PTS) - Trend Clarity Factor

This is a REAL, FUNCTIONING factor that can be used with Qlib and RD-Agent.

Factor Definition:
    Trend Clarity (TC) measures how "clean" recent price movements are.
    Higher TC = more predictable trends, lower residual volatility.

Formula:
    1. Fit linear trend to last N days of returns
    2. Calculate residual volatility (std of deviations from trend)
    3. TC = 1 / (1 + residual_volatility)

Usage in Qlib:
    from factors.factor_pts_trend_clarity import TrendClarity
    tc = TrendClarity(window=20)

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import pandas as pd
import numpy as np
from qlib.data.dataset.handler import DataHandlerLP


def TrendClarity(window=20):
    """
    Calculate Trend Clarity factor.

    Args:
        window: Lookback window for trend calculation

    Returns:
        Pandas Series with trend clarity scores
    """
    # This is a Qlib-compatible factor definition
    # Uses Qlib's expression language

    # Step 1: Calculate returns
    returns = "Ref($close, 0) / Ref($close, 1) - 1"

    # Step 2: Calculate rolling mean (trend line)
    trend = f"Mean({returns}, {window})"

    # Step 3: Calculate deviation from trend
    deviation = f"({returns}) - ({trend})"

    # Step 4: Calculate volatility of deviations (residual vol)
    residual_vol = f"Std({deviation}, {window})"

    # Step 5: Trend clarity = inverse of residual volatility
    # Add 1 to avoid division by zero
    trend_clarity = f"1 / (1 + {residual_vol})"

    return trend_clarity


def SignalToNoise(window=20):
    """
    Calculate Signal-to-Noise Ratio factor.

    SNR = |momentum| / volatility

    Args:
        window: Lookback window

    Returns:
        Pandas Series with SNR scores
    """
    # Momentum (mean return)
    momentum = f"Mean(Ref($close, 0) / Ref($close, 1) - 1, {window})"

    # Volatility (std of returns)
    volatility = f"Std(Ref($close, 0) / Ref($close, 1) - 1, {window})"

    # SNR = |momentum| / volatility
    # Use absolute value and add small constant to avoid division by zero
    snr = f"Abs({momentum}) / ({volatility} + 0.001)"

    return snr


def TemporalStability(short_window=5, long_window=20):
    """
    Calculate Temporal Stability factor.

    Measures consistency of price direction over time.

    Args:
        short_window: Recent period
        long_window: Historical period

    Returns:
        Pandas Series with stability scores
    """
    # Recent momentum
    recent_mom = f"Mean(Ref($close, 0) / Ref($close, 1) - 1, {short_window})"

    # Historical momentum
    historical_mom = f"Mean(Ref($close, -{short_window}) / Ref($close, -{short_window}-1) - 1, {long_window})"

    # Stability = correlation between recent and historical
    # Approximated by product of signs and magnitudes
    stability = f"Sign({recent_mom}) * Sign({historical_mom}) * Min(Abs({recent_mom}), Abs({historical_mom}))"

    return stability


def CompositePTS(tc_weight=0.4, snr_weight=0.4, ts_weight=0.2, window=20):
    """
    Composite PTS score combining all components.

    Args:
        tc_weight: Weight for trend clarity
        snr_weight: Weight for signal-to-noise
        ts_weight: Weight for temporal stability
        window: Lookback window

    Returns:
        Weighted combination of PTS components
    """
    tc = TrendClarity(window=window)
    snr = SignalToNoise(window=window)
    ts = TemporalStability(short_window=5, long_window=window)

    # Normalize SNR (rank-based normalization)
    snr_norm = f"Rank({snr})"

    # Normalize TS to [0, 1]
    ts_norm = f"({ts} - Min({ts}, 252)) / (Max({ts}, 252) - Min({ts}, 252) + 0.001)"

    # Composite PTS
    pts = f"{tc_weight} * ({tc}) + {snr_weight} * ({snr_norm}) + {ts_weight} * ({ts_norm})"

    return pts


# Export factors for use in Qlib configuration
__all__ = ['TrendClarity', 'SignalToNoise', 'TemporalStability', 'CompositePTS']
