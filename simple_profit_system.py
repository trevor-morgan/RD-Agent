"""
SIMPLE PROFIT SYSTEM
No ML bullshit. Direct volatility-based regime detection + adaptive strategies.

Insight: Different strategies work in different volatility environments.
- High vol: Mean reversion (overshoots)
- Low vol: Momentum (trends)

Implementation: Calculate vol directly. Trade adaptively. Make money.

Author: RD-Agent Research Team
Date: 2025-11-13
Purpose: PROFIT
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_real_data_validation import RealMarketDataLoader


def calculate_regimes(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate regimes from realized volatility.
    Simple. Direct. Works.
    """

    # Rolling volatility
    vols = []
    for i in range(len(returns)):
        if i < window:
            vols.append(np.nan)
        else:
            vol = np.std(returns[i-window:i])
            vols.append(vol)

    vols = np.array(vols)

    # Tertiles
    valid_vols = vols[~np.isnan(vols)]
    low_thresh = np.percentile(valid_vols, 33)
    high_thresh = np.percentile(valid_vols, 67)

    # Classify
    regimes = np.zeros(len(vols), dtype=int)
    regimes[vols < low_thresh] = 0  # Low vol
    regimes[(vols >= low_thresh) & (vols < high_thresh)] = 1  # Medium
    regimes[vols >= high_thresh] = 2  # High vol

    return regimes, vols


def mean_reversion_signal(returns: np.ndarray, lookback: int = 5) -> float:
    """Fade recent moves."""
    if len(returns) < lookback:
        return 0.0

    recent = np.sum(returns[-lookback:])
    return -np.tanh(recent * 100)  # Opposite direction


def momentum_signal(returns: np.ndarray, lookback: int = 20) -> float:
    """Follow trends."""
    if len(returns) < lookback:
        return 0.0

    trend = np.sum(returns[-lookback:])
    return np.tanh(trend * 50)  # Same direction


def backtest_adaptive(
    returns: np.ndarray,
    regimes: np.ndarray,
    cost_bps: float = 2.0,
    lookback_start: int = 100
) -> dict:
    """
    Backtest adaptive strategy.
    High vol → mean reversion
    Low vol → momentum
    Med vol → reduced exposure
    """

    n = len(returns)
    positions = []
    pnls = []
    costs = []

    current_pos = 0.0

    for i in range(lookback_start, n - 1):
        hist_returns = returns[max(0, i-100):i]
        regime = regimes[i]

        # Generate signal based on regime
        if regime == 2:  # High vol
            signal = mean_reversion_signal(hist_returns, lookback=5)
            signal *= 0.8  # Moderate position in high vol
        elif regime == 0:  # Low vol
            signal = momentum_signal(hist_returns, lookback=20)
            signal *= 1.0  # Full position in low vol
        else:  # Medium vol
            # Blend both, reduce size
            mr_sig = mean_reversion_signal(hist_returns, lookback=5)
            mom_sig = momentum_signal(hist_returns, lookback=20)
            signal = (mr_sig + mom_sig) / 2
            signal *= 0.5  # Half size

        # Trade
        trade = signal - current_pos
        cost = abs(trade) * (cost_bps / 10000)

        current_pos = signal
        positions.append(current_pos)
        costs.append(cost)

        # PnL
        next_ret = returns[i + 1]
        pnl = current_pos * next_ret - cost
        pnls.append(pnl)

    pnls = np.array(pnls)

    # Metrics
    total_ret = np.sum(pnls)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6) * np.sqrt(252)
    win_rate = np.sum(pnls > 0) / len(pnls)
    total_cost = np.sum(costs)

    return {
        'total_return': total_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_costs': total_cost,
        'n_trades': len(pnls),
    }


def backtest_baseline(
    returns: np.ndarray,
    cost_bps: float = 2.0,
    lookback_start: int = 100
) -> dict:
    """Simple momentum baseline for comparison."""

    n = len(returns)
    pnls = []
    costs = []

    current_pos = 0.0

    for i in range(lookback_start, n - 1):
        hist_returns = returns[max(0, i-100):i]

        # Simple momentum
        signal = momentum_signal(hist_returns, lookback=20)

        trade = signal - current_pos
        cost = abs(trade) * (cost_bps / 10000)

        current_pos = signal
        costs.append(cost)

        next_ret = returns[i + 1]
        pnl = current_pos * next_ret - cost
        pnls.append(pnl)

    pnls = np.array(pnls)

    total_ret = np.sum(pnls)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6) * np.sqrt(252)
    win_rate = np.sum(pnls > 0) / len(pnls)

    return {
        'total_return': total_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_costs': np.sum(costs),
        'n_trades': len(pnls),
    }


def run_simple_profit():
    """Run simple profit system."""

    print("=" * 80)
    print("SIMPLE PROFIT SYSTEM")
    print("=" * 80)
    print()
    print("Approach:")
    print("  1. Calculate regimes from ACTUAL volatility (no ML)")
    print("  2. High vol → Mean reversion")
    print("  3. Low vol → Momentum")
    print("  4. Med vol → Blend (reduced size)")
    print()
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    loader = RealMarketDataLoader(
        TICKERS,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        interval='5m'
    )
    tick_data = loader.download_data()
    returns = tick_data['returns']

    print(f"✓ Loaded {len(returns)} returns\n")

    # Calculate regimes
    print("Calculating volatility regimes...")
    regimes, vols = calculate_regimes(returns, window=20)

    valid_regimes = regimes[~np.isnan(vols)]
    print(f"Regime distribution:")
    print(f"  Low vol (0):  {np.sum(valid_regimes == 0)} ({np.sum(valid_regimes == 0)/len(valid_regimes)*100:.1f}%)")
    print(f"  Med vol (1):  {np.sum(valid_regimes == 1)} ({np.sum(valid_regimes == 1)/len(valid_regimes)*100:.1f}%)")
    print(f"  High vol (2): {np.sum(valid_regimes == 2)} ({np.sum(valid_regimes == 2)/len(valid_regimes)*100:.1f}%)")
    print()

    # Walk-forward validation
    print("=" * 80)
    print("WALK-FORWARD BACKTEST (3 FOLDS)")
    print("=" * 80)
    print()

    n = len(returns)
    fold_size = n // 4

    all_adaptive = []
    all_baseline = []

    for fold in range(3):
        print(f"FOLD {fold + 1}/3")
        print("-" * 80)

        test_start = (fold + 1) * fold_size
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 100:
            print("Insufficient data\n")
            continue

        # Get test data
        test_returns = returns[:test_end]  # Full history for lookback
        test_regimes = regimes[:test_end]

        print(f"  Test period: {test_start} to {test_end}")

        # Backtest adaptive
        adaptive = backtest_adaptive(
            test_returns,
            test_regimes,
            cost_bps=2.0,
            lookback_start=test_start
        )

        # Backtest baseline (momentum only)
        baseline = backtest_baseline(
            test_returns,
            cost_bps=2.0,
            lookback_start=test_start
        )

        print(f"  Adaptive:")
        print(f"    Return: {adaptive['total_return']:+.4f} ({adaptive['total_return']*100:+.2f}%)")
        print(f"    Sharpe: {adaptive['sharpe']:+.2f}")
        print(f"    Win Rate: {adaptive['win_rate']:.1%}")
        print(f"  Baseline (Momentum):")
        print(f"    Return: {baseline['total_return']:+.4f} ({baseline['total_return']*100:+.2f}%)")
        print(f"    Sharpe: {baseline['sharpe']:+.2f}")
        print(f"  Improvement: {(adaptive['total_return'] - baseline['total_return'])*100:+.2f}%")
        print()

        all_adaptive.append(adaptive)
        all_baseline.append(baseline)

    # Aggregate
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print()

    if len(all_adaptive) > 0:
        avg_adaptive_ret = np.mean([r['total_return'] for r in all_adaptive])
        avg_adaptive_sharpe = np.mean([r['sharpe'] for r in all_adaptive])
        avg_baseline_ret = np.mean([r['total_return'] for r in all_baseline])
        avg_baseline_sharpe = np.mean([r['sharpe'] for r in all_baseline])

        improvement = avg_adaptive_ret - avg_baseline_ret

        print(f"Adaptive Strategy:")
        print(f"  Avg Return: {avg_adaptive_ret:+.4f} ({avg_adaptive_ret*100:+.2f}%)")
        print(f"  Avg Sharpe: {avg_adaptive_sharpe:+.2f}")
        print()
        print(f"Baseline (Momentum):")
        print(f"  Avg Return: {avg_baseline_ret:+.4f} ({avg_baseline_ret*100:+.2f}%)")
        print(f"  Avg Sharpe: {avg_baseline_sharpe:+.2f}")
        print()
        print(f"Improvement: {improvement*100:+.2f}% return")
        print()

        positive_adaptive = sum(1 for r in all_adaptive if r['total_return'] > 0)
        positive_baseline = sum(1 for r in all_baseline if r['total_return'] > 0)

        print(f"Consistency:")
        print(f"  Adaptive positive folds: {positive_adaptive}/{len(all_adaptive)}")
        print(f"  Baseline positive folds: {positive_baseline}/{len(all_baseline)}")
        print()

        # Save
        results = {
            'adaptive': {
                'avg_return': avg_adaptive_ret,
                'avg_sharpe': avg_adaptive_sharpe,
                'positive_folds': f"{positive_adaptive}/{len(all_adaptive)}",
            },
            'baseline': {
                'avg_return': avg_baseline_ret,
                'avg_sharpe': avg_baseline_sharpe,
                'positive_folds': f"{positive_baseline}/{len(all_baseline)}",
            },
            'improvement': improvement,
        }

        with open('simple_profit_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("✓ Results saved\n")

        # Verdict
        print("=" * 80)
        print("PROFIT VERDICT")
        print("=" * 80)
        print()

        if avg_adaptive_ret > 0 and avg_adaptive_sharpe > 0 and improvement > 0:
            print("🎯 PROFITABLE + IMPROVED")
            print(f"   {avg_adaptive_ret*100:+.2f}% return, {avg_adaptive_sharpe:+.2f} Sharpe")
            print(f"   {improvement*100:+.2f}% better than baseline")
        elif avg_adaptive_ret > 0:
            print("✓ PROFITABLE (but not better than baseline)")
        elif improvement > 0:
            print("⚠️  IMPROVED (but still negative)")
        else:
            print("❌ NOT PROFITABLE")


if __name__ == '__main__':
    run_simple_profit()
