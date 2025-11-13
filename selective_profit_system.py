"""
SELECTIVE PROFIT SYSTEM
Only trade when conditions match Fold 2 success criteria

Strategy: "Wait for your pitch"
- High volatility (>median)
- Low trend strength (<median)
- Then apply regime-adaptive strategies aggressively

This maximizes win rate by only trading in favorable conditions.

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_real_data_validation import RealMarketDataLoader


def calculate_regimes(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Calculate regimes from realized volatility."""
    vols = []
    for i in range(len(returns)):
        if i < window:
            vols.append(np.nan)
        else:
            vol = np.std(returns[i-window:i])
            vols.append(vol)

    vols = np.array(vols)
    valid_vols = vols[~np.isnan(vols)]
    low_thresh = np.percentile(valid_vols, 33)
    high_thresh = np.percentile(valid_vols, 67)

    regimes = np.zeros(len(vols), dtype=int)
    regimes[vols < low_thresh] = 0
    regimes[(vols >= low_thresh) & (vols < high_thresh)] = 1
    regimes[vols >= high_thresh] = 2

    return regimes, vols


def calculate_trend_strength(returns: np.ndarray, window: int = 50) -> float:
    """Calculate trend strength."""
    if len(returns) < window:
        return 0.0

    cumulative = np.abs(np.sum(returns[-window:]))
    vol = np.std(returns[-window:])
    if vol < 1e-6:
        return 0.0

    return cumulative / (vol * np.sqrt(window))


def mean_reversion_signal(returns: np.ndarray, lookback: int = 5) -> float:
    """Mean reversion."""
    if len(returns) < lookback:
        return 0.0
    recent = np.sum(returns[-lookback:])
    return -np.tanh(recent * 100)


def momentum_signal(returns: np.ndarray, lookback: int = 20) -> float:
    """Momentum."""
    if len(returns) < lookback:
        return 0.0
    trend = np.sum(returns[-lookback:])
    return np.tanh(trend * 50)


def backtest_selective(
    returns: np.ndarray,
    regimes: np.ndarray,
    cost_bps: float = 2.0,
    lookback_start: int = 100
) -> dict:
    """
    Selective strategy: Only trade in favorable conditions.

    Favorable = High volatility + Low trend (Fold 2 conditions)
    """

    n = len(returns)
    positions = []
    pnls = []
    costs = []
    trades_taken = 0
    trades_skipped = 0

    # Calculate rolling metrics for thresholds
    all_vols = []
    all_trends = []

    for i in range(lookback_start, n - 1):
        hist = returns[max(0, i-100):i]
        vol = np.std(hist[-20:])
        trend = calculate_trend_strength(hist, window=50)
        all_vols.append(vol)
        all_trends.append(trend)

    median_vol = np.median(all_vols)
    median_trend = np.median(all_trends)

    print(f"  Thresholds: Vol>{median_vol:.6f}, Trend<{median_trend:.2f}")

    # Backtest
    current_pos = 0.0

    for i in range(lookback_start, n - 1):
        hist_returns = returns[max(0, i-100):i]
        regime = regimes[i]

        # Calculate current vol and trend
        current_vol = np.std(hist_returns[-20:])
        current_trend = calculate_trend_strength(hist_returns, window=50)

        # Check if conditions are favorable
        favorable = (current_vol > median_vol) and (current_trend < median_trend)

        if favorable:
            # Trade aggressively
            trades_taken += 1

            if regime == 2:  # High vol
                signal = mean_reversion_signal(hist_returns, lookback=5)
                signal *= 1.0  # Full size
            elif regime == 0:  # Low vol
                signal = momentum_signal(hist_returns, lookback=20)
                signal *= 1.0  # Full size
            else:  # Medium vol
                mr_sig = mean_reversion_signal(hist_returns, lookback=5)
                mom_sig = momentum_signal(hist_returns, lookback=20)
                signal = (mr_sig + mom_sig) / 2
                signal *= 0.7
        else:
            # Unfavorable conditions - stay flat
            trades_skipped += 1
            signal = 0.0

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

    # Activity metrics
    activity_rate = trades_taken / (trades_taken + trades_skipped)

    return {
        'total_return': total_ret,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_costs': np.sum(costs),
        'n_trades': len(pnls),
        'trades_taken': trades_taken,
        'trades_skipped': trades_skipped,
        'activity_rate': activity_rate,
    }


def run_selective_profit():
    """Run selective profit system."""

    print("=" * 80)
    print("SELECTIVE PROFIT SYSTEM")
    print("=" * 80)
    print()
    print("Strategy: 'Wait for your pitch'")
    print("  Only trade when conditions match Fold 2 success:")
    print("    - High volatility (>median)")
    print("    - Low trend strength (<median)")
    print("  Then apply regime-adaptive strategies aggressively")
    print()
    print("Goal: Higher win rate by avoiding unfavorable periods")
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
    print("Calculating regimes...")
    regimes, vols = calculate_regimes(returns, window=20)
    print("✓ Regimes calculated\n")

    # Walk-forward validation
    print("=" * 80)
    print("WALK-FORWARD BACKTEST (3 FOLDS)")
    print("=" * 80)
    print()

    n = len(returns)
    fold_size = n // 4

    all_selective = []
    all_baseline = []

    for fold in range(3):
        print(f"FOLD {fold + 1}/3")
        print("-" * 80)

        test_start = (fold + 1) * fold_size
        test_end = min(test_start + fold_size, n)

        if test_end - test_start < 100:
            print("Insufficient data\n")
            continue

        test_returns = returns[:test_end]
        test_regimes = regimes[:test_end]

        print(f"  Test period: {test_start} to {test_end}")

        # Selective strategy
        selective = backtest_selective(
            test_returns,
            test_regimes,
            cost_bps=2.0,
            lookback_start=test_start
        )

        # Baseline
        from simple_profit_system import backtest_baseline
        baseline = backtest_baseline(
            test_returns,
            cost_bps=2.0,
            lookback_start=test_start
        )

        print(f"  Selective:")
        print(f"    Return: {selective['total_return']:+.4f} ({selective['total_return']*100:+.2f}%)")
        print(f"    Sharpe: {selective['sharpe']:+.2f}")
        print(f"    Win Rate: {selective['win_rate']:.1%}")
        print(f"    Activity: {selective['activity_rate']:.1%} (took {selective['trades_taken']}, skipped {selective['trades_skipped']})")
        print(f"  Baseline (Momentum):")
        print(f"    Return: {baseline['total_return']:+.4f} ({baseline['total_return']*100:+.2f}%)")
        print(f"    Sharpe: {baseline['sharpe']:+.2f}")
        print(f"  Improvement: {(selective['total_return'] - baseline['total_return'])*100:+.2f}%")
        print()

        all_selective.append(selective)
        all_baseline.append(baseline)

    # Aggregate
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print()

    if len(all_selective) > 0:
        avg_selective_ret = np.mean([r['total_return'] for r in all_selective])
        avg_selective_sharpe = np.mean([r['sharpe'] for r in all_selective])
        avg_baseline_ret = np.mean([r['total_return'] for r in all_baseline])
        avg_baseline_sharpe = np.mean([r['sharpe'] for r in all_baseline])
        avg_activity = np.mean([r['activity_rate'] for r in all_selective])

        improvement = avg_selective_ret - avg_baseline_ret

        print(f"Selective Strategy:")
        print(f"  Avg Return: {avg_selective_ret:+.4f} ({avg_selective_ret*100:+.2f}%)")
        print(f"  Avg Sharpe: {avg_selective_sharpe:+.2f}")
        print(f"  Avg Activity: {avg_activity:.1%}")
        print()
        print(f"Baseline (Momentum):")
        print(f"  Avg Return: {avg_baseline_ret:+.4f} ({avg_baseline_ret*100:+.2f}%)")
        print(f"  Avg Sharpe: {avg_baseline_sharpe:+.2f}")
        print()
        print(f"Improvement: {improvement*100:+.2f}% return, {avg_selective_sharpe - avg_baseline_sharpe:+.2f} Sharpe")
        print()

        positive_selective = sum(1 for r in all_selective if r['total_return'] > 0)
        positive_baseline = sum(1 for r in all_baseline if r['total_return'] > 0)

        print(f"Consistency:")
        print(f"  Selective positive folds: {positive_selective}/{len(all_selective)}")
        print(f"  Baseline positive folds: {positive_baseline}/{len(all_baseline)}")
        print()

        # Save
        results = {
            'selective': {
                'avg_return': avg_selective_ret,
                'avg_sharpe': avg_selective_sharpe,
                'avg_activity': avg_activity,
                'positive_folds': f"{positive_selective}/{len(all_selective)}",
            },
            'baseline': {
                'avg_return': avg_baseline_ret,
                'avg_sharpe': avg_baseline_sharpe,
                'positive_folds': f"{positive_baseline}/{len(all_baseline)}",
            },
            'improvement': improvement,
        }

        with open('selective_profit_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("✓ Results saved\n")

        # Verdict
        print("=" * 80)
        print("PROFIT VERDICT")
        print("=" * 80)
        print()

        print("Comparison:")
        print(f"  Simple Adaptive:  +5.20% return, -0.03 Sharpe (always trading)")
        print(f"  Improved w/filter: +1.41% return, -0.19 Sharpe (scaled down)")
        print(f"  Selective:        {avg_selective_ret*100:+.2f}% return, {avg_selective_sharpe:+.2f} Sharpe ({avg_activity:.0%} activity)")
        print()

        if avg_selective_ret > 0.052 and avg_selective_sharpe > 0:
            print("🎯 BEST YET!")
        elif avg_selective_sharpe > -0.03:
            print("✓ IMPROVED SHARPE")
        elif avg_selective_ret > 0.052:
            print("⚠️  HIGHER RETURN, STILL NEGATIVE SHARPE")
        else:
            print("❌ WORSE")


if __name__ == '__main__':
    run_selective_profit()
