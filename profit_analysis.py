"""
PROFIT ANALYSIS
Deep dive into why Fold 2 worked (+81%) and Folds 1,3 failed
Find the conditions for profitability

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


def analyze_fold_characteristics():
    """Analyze what made each fold different."""

    print("=" * 80)
    print("PROFIT ANALYSIS")
    print("=" * 80)
    print()
    print("Goal: Understand why Fold 2 worked (+81%) and Folds 1,3 failed")
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
    returns_raw = tick_data['returns']

    # Average across assets to get "market" return
    if len(returns_raw.shape) == 2:
        returns = np.mean(returns_raw, axis=1)
    else:
        returns = returns_raw

    print(f"✓ Loaded {len(returns)} returns\n")

    # Calculate regimes
    print("Calculating regimes...")

    def calculate_regimes(returns: np.ndarray, window: int = 20) -> np.ndarray:
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

    regimes, vols = calculate_regimes(returns, window=20)
    print("✓ Regimes calculated\n")

    # Analyze each fold
    n = len(returns)
    fold_size = n // 4

    print("=" * 80)
    print("FOLD ANALYSIS")
    print("=" * 80)
    print()

    analyses = []

    for fold in range(3):
        print(f"FOLD {fold + 1}/3")
        print("-" * 80)

        test_start = (fold + 1) * fold_size
        test_end = min(test_start + fold_size, n)

        # Get fold data
        fold_returns = returns[test_start:test_end]
        fold_regimes = regimes[test_start:test_end]
        fold_vols = vols[test_start:test_end]

        # Remove NaN
        valid_idx = ~np.isnan(fold_vols)
        fold_returns_clean = fold_returns[valid_idx]
        fold_regimes_clean = fold_regimes[valid_idx]
        fold_vols_clean = fold_vols[valid_idx]

        print(f"  Period: {test_start} to {test_end} ({len(fold_returns_clean)} valid samples)")
        print()

        # Market characteristics
        mean_ret = np.mean(fold_returns_clean)
        vol = np.std(fold_returns_clean)
        skew = pd.Series(fold_returns_clean).skew()
        kurt = pd.Series(fold_returns_clean).kurtosis()

        # Trend strength
        cumulative = np.cumsum(fold_returns_clean)
        trend_strength = abs(cumulative[-1])

        # Regime distribution
        n_low = np.sum(fold_regimes_clean == 0)
        n_med = np.sum(fold_regimes_clean == 1)
        n_high = np.sum(fold_regimes_clean == 2)

        # Mean reversion opportunities (high vol periods)
        high_vol_returns = fold_returns_clean[fold_regimes_clean == 2]
        mr_opportunity = 0.0
        if len(high_vol_returns) > 5:
            # Check if high vol returns reverse
            for i in range(len(high_vol_returns) - 5):
                move = np.sum(high_vol_returns[i:i+3])
                reversal = np.sum(high_vol_returns[i+3:i+5])
                if move * reversal < 0:  # Opposite signs
                    mr_opportunity += 1
            mr_opportunity = mr_opportunity / max(len(high_vol_returns) - 5, 1)

        # Momentum opportunities (low vol periods)
        low_vol_returns = fold_returns_clean[fold_regimes_clean == 0]
        mom_opportunity = 0.0
        if len(low_vol_returns) > 10:
            # Check if low vol trends continue
            for i in range(len(low_vol_returns) - 10):
                past_trend = np.sum(low_vol_returns[i:i+5])
                future_trend = np.sum(low_vol_returns[i+5:i+10])
                if past_trend * future_trend > 0:  # Same signs
                    mom_opportunity += 1
            mom_opportunity = mom_opportunity / max(len(low_vol_returns) - 10, 1)

        print(f"  Market Characteristics:")
        print(f"    Mean return:     {mean_ret:+.6f} ({mean_ret*100:+.4f}%)")
        print(f"    Volatility:      {vol:.6f} ({vol*100:.4f}%)")
        print(f"    Skewness:        {skew:+.2f}")
        print(f"    Kurtosis:        {kurt:+.2f}")
        print(f"    Trend strength:  {trend_strength:.4f}")
        print()

        print(f"  Regime Distribution:")
        print(f"    Low vol (0):  {n_low} ({n_low/len(fold_regimes_clean)*100:.1f}%)")
        print(f"    Med vol (1):  {n_med} ({n_med/len(fold_regimes_clean)*100:.1f}%)")
        print(f"    High vol (2): {n_high} ({n_high/len(fold_regimes_clean)*100:.1f}%)")
        print()

        print(f"  Strategy Opportunities:")
        print(f"    Mean reversion (high vol): {mr_opportunity:.1%}")
        print(f"    Momentum (low vol):        {mom_opportunity:.1%}")
        print()

        # Extreme events
        extreme_threshold = np.percentile(np.abs(fold_returns_clean), 95)
        n_extreme = np.sum(np.abs(fold_returns_clean) > extreme_threshold)

        print(f"  Risk Metrics:")
        print(f"    Extreme events (>95th pct): {n_extreme} ({n_extreme/len(fold_returns_clean)*100:.1f}%)")
        print(f"    Max drawdown: {np.min(cumulative):+.4f}")
        print(f"    Max runup:    {np.max(cumulative):+.4f}")
        print()

        analyses.append({
            'fold': fold + 1,
            'mean_return': mean_ret,
            'volatility': vol,
            'skewness': skew,
            'kurtosis': kurt,
            'trend_strength': trend_strength,
            'regime_low_pct': n_low / len(fold_regimes_clean),
            'regime_med_pct': n_med / len(fold_regimes_clean),
            'regime_high_pct': n_high / len(fold_regimes_clean),
            'mr_opportunity': mr_opportunity,
            'mom_opportunity': mom_opportunity,
            'n_extreme': n_extreme,
        })

    # Compare folds
    print("=" * 80)
    print("FOLD COMPARISON")
    print("=" * 80)
    print()

    df = pd.DataFrame(analyses)
    print("Key differences:")
    print()

    # Fold 2 had +81% improvement, compare to others
    fold2 = df[df['fold'] == 2].iloc[0]
    fold1 = df[df['fold'] == 1].iloc[0]
    fold3 = df[df['fold'] == 3].iloc[0]

    print("Fold 2 (WINNER +81%) vs Fold 1 (-4.56%) vs Fold 3 (-64%):")
    print()

    for col in df.columns:
        if col == 'fold':
            continue
        print(f"  {col}:")
        print(f"    Fold 1: {fold1[col]:+.4f}")
        print(f"    Fold 2: {fold2[col]:+.4f} ← WINNER")
        print(f"    Fold 3: {fold3[col]:+.4f}")
        print()

    # Identify key differentiators
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    # What made Fold 2 special?
    if fold2['mr_opportunity'] > fold1['mr_opportunity'] and fold2['mr_opportunity'] > fold3['mr_opportunity']:
        print("✓ Fold 2 had BEST mean reversion opportunities")
        print(f"  Fold 2: {fold2['mr_opportunity']:.1%}")
        print(f"  Fold 1: {fold1['mr_opportunity']:.1%}")
        print(f"  Fold 3: {fold3['mr_opportunity']:.1%}")
        print()

    if fold2['mom_opportunity'] > fold1['mom_opportunity'] and fold2['mom_opportunity'] > fold3['mom_opportunity']:
        print("✓ Fold 2 had BEST momentum opportunities")
        print(f"  Fold 2: {fold2['mom_opportunity']:.1%}")
        print(f"  Fold 1: {fold1['mom_opportunity']:.1%}")
        print(f"  Fold 3: {fold3['mom_opportunity']:.1%}")
        print()

    if abs(fold2['mean_return']) < abs(fold1['mean_return']) or abs(fold2['mean_return']) < abs(fold3['mean_return']):
        print("✓ Fold 2 had more BALANCED market (less one-directional)")
        print(f"  Fold 2: {fold2['mean_return']:+.6f}")
        print(f"  Fold 1: {fold1['mean_return']:+.6f}")
        print(f"  Fold 3: {fold3['mean_return']:+.6f}")
        print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("To capture Fold 2's success while avoiding Fold 1/3's failures:")
    print()

    if fold2['mr_opportunity'] > 0.3:
        print("1. STRENGTHEN mean reversion in high vol")
        print("   - Current: -tanh(move * 50)")
        print("   - Try: Stronger fade, faster entry/exit")
        print()

    if fold2['mom_opportunity'] > 0.4:
        print("2. STRENGTHEN momentum in low vol")
        print("   - Current: tanh(trend * 30)")
        print("   - Try: Longer lookback, stronger position")
        print()

    if fold2['regime_high_pct'] > 0.35:
        print("3. INCREASE exposure in high vol regimes")
        print(f"   - Fold 2 had {fold2['regime_high_pct']:.1%} high vol")
        print("   - Mean reversion works best there")
        print()

    if fold3['trend_strength'] > fold2['trend_strength']:
        print("4. ADD trend filter to avoid strong one-way markets")
        print(f"   - Fold 3 (failed): trend strength {fold3['trend_strength']:.4f}")
        print(f"   - Fold 2 (won): trend strength {fold2['trend_strength']:.4f}")
        print("   - Reduce size when cumulative move > threshold")
        print()

    # Save
    with open('profit_analysis_results.json', 'w') as f:
        json.dump({
            'fold_characteristics': analyses,
            'winner': 'Fold 2',
            'winner_return_improvement': '+81.52%',
        }, f, indent=2, default=str)

    print("✓ Analysis saved to: profit_analysis_results.json")


if __name__ == '__main__':
    analyze_fold_characteristics()
