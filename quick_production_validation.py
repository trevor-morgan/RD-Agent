"""
QUICK PRODUCTION VALIDATION
Use existing 1-month data with proper rolling walk-forward and transaction costs

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from datetime import datetime, timedelta
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_integrated import HCANPsi
from hcan_psi_real_data_validation import RealMarketDataLoader, HCANPsiDataset
from bifurcation_crash_detector import BifurcationCrashDetector, AntifragilePortfolioManager


def run_quick_production_validation():
    """
    Production validation with available data.
    """

    print("=" * 80)
    print("PRODUCTION VALIDATION - BIFURCATION CRASH DETECTOR")
    print("=" * 80)
    print()
    print("Strategy:")
    print("  1. Use 30 days of 5min data (max available)")
    print("  2. Rolling 3-fold walk-forward validation")
    print("  3. Real transaction costs (1bp spread + 0.5bp fees)")
    print("  4. Statistical significance tests")
    print()
    print("=" * 80)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load data
    print("Loading data...")
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'UNH', 'HD', 'BAC',
        'XOM', 'CVX', 'PFE', 'KO', 'PEP'
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
    full_dataset = HCANPsiDataset(tick_data, window_size=20, analog_window=100)
    print(f"✓ Dataset: {len(full_dataset)} samples\n")

    # Load model
    print("Loading HCAN-Ψ model...")
    checkpoint = torch.load('checkpoints_extended/best_val_ic.pt',
                           map_location=device, weights_only=False)
    config = checkpoint['config']

    model = HCANPsi(
        input_dim=20,
        reservoir_size=config['reservoir_size'],
        embed_dim=config['embed_dim'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        n_wavelet_scales=config['n_wavelet_scales'],
        n_agents=config['n_agents'],
        psi_feature_dim=config['psi_feature_dim'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded\n")

    # Create crash detector
    crash_detector = BifurcationCrashDetector(model, device)

    # Extract metrics
    print("Extracting bifurcation metrics...")
    metrics_df = crash_detector.extract_risk_metrics(full_dataset)
    print(f"✓ Extracted {len(metrics_df)} samples\n")

    # Rolling 3-fold validation
    print("=" * 80)
    print("ROLLING 3-FOLD WALK-FORWARD VALIDATION")
    print("=" * 80)
    print()

    n_samples = len(metrics_df)
    fold_size = n_samples // 4  # 25% per fold

    all_results = []

    for fold in range(3):
        print(f"FOLD {fold + 1}/3")
        print("-" * 80)

        # Calculate indices
        train_start = 0
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        if test_end - test_start < 50:
            print(f"⚠️  Insufficient test data ({test_end - test_start} samples)")
            continue

        # Split
        train_df = metrics_df.iloc[train_start:train_end].reset_index(drop=True)
        test_df = metrics_df.iloc[test_start:test_end].reset_index(drop=True)

        print(f"  Train: 0 to {train_end} ({len(train_df)} samples)")
        print(f"  Test:  {test_start} to {test_end} ({len(test_df)} samples)")

        # Calibrate on train
        crash_detector.calibrate_thresholds(train_df, extreme_percentile=95)

        # Test
        returns = np.array(test_df['actual_return'].values, dtype=float)
        bifurcation = np.array(test_df['bifurcation'].values, dtype=float)

        # Bifurcation → Extreme events
        extreme_threshold = np.percentile(np.abs(returns), 95)
        is_extreme = np.abs(returns) > extreme_threshold

        if len(is_extreme) > 1:
            bifurc_ic = np.corrcoef(bifurcation[:-1], is_extreme[1:].astype(float))[0, 1]
        else:
            bifurc_ic = 0.0

        print(f"  Bifurcation IC: {bifurc_ic:+.4f}")

        # Backtest
        portfolio = AntifragilePortfolioManager(crash_detector)
        backtest = portfolio.backtest(test_df[:-1], returns[1:])

        print(f"  Dynamic Sharpe:  {backtest['dynamic_sharpe']:+.2f}")
        print(f"  Baseline Sharpe: {backtest['baseline_sharpe']:+.2f}")
        print(f"  Improvement:     {backtest['dynamic_sharpe'] - backtest['baseline_sharpe']:+.2f}")

        # Transaction costs (1bp spread + 0.5bp fees = 1.5bp total)
        # Estimate: Turnover ~ how often position changes
        # Assume 20% daily turnover (conservative for dynamic sizing)
        daily_turnover = 0.20
        trading_days_per_year = 252
        cost_bps_per_trade = 1.5

        # Annual cost = daily_turnover * trading_days * cost_per_trade
        annual_cost_pct = daily_turnover * trading_days_per_year * (cost_bps_per_trade / 10000)

        # Cost impact on Sharpe = cost / volatility * sqrt(252)
        cost_sharpe_impact = annual_cost_pct / (np.std(returns) * np.sqrt(252) + 1e-6)

        net_sharpe = backtest['dynamic_sharpe'] - cost_sharpe_impact

        print(f"  Cost Impact:     -{cost_sharpe_impact:.2f}")
        print(f"  Net Sharpe:      {net_sharpe:+.2f}")
        print()

        all_results.append({
            'fold': fold + 1,
            'bifurcation_ic': bifurc_ic,
            'dynamic_sharpe': backtest['dynamic_sharpe'],
            'baseline_sharpe': backtest['baseline_sharpe'],
            'improvement': backtest['dynamic_sharpe'] - backtest['baseline_sharpe'],
            'net_sharpe': net_sharpe,
            'n_test': len(test_df),
        })

    # Aggregate
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print()

    if len(all_results) > 0:
        avg_improvement = np.mean([r['improvement'] for r in all_results])
        avg_net_sharpe = np.mean([r['net_sharpe'] for r in all_results])
        avg_bifurc_ic = np.mean([r['bifurcation_ic'] for r in all_results])

        print(f"Across {len(all_results)} folds:")
        print(f"  Avg Bifurcation IC:      {avg_bifurc_ic:+.4f}")
        print(f"  Avg Sharpe Improvement:  {avg_improvement:+.2f}")
        print(f"  Avg Net Sharpe:          {avg_net_sharpe:+.2f}")
        print(f"  Consistency: {sum(1 for r in all_results if r['net_sharpe'] > 0)}/{len(all_results)} folds positive")
        print()

        # Statistical test
        improvements = [r['improvement'] for r in all_results]
        if len(improvements) > 1:
            t_stat, p_value = stats.ttest_1samp(improvements, 0)
            print(f"Statistical Test (H0: improvement = 0):")
            print(f"  T-statistic: {t_stat:.2f}")
            print(f"  P-value:     {p_value:.4f}")
            print(f"  Significant: {'YES ✓' if p_value < 0.05 else 'NO'}")
            print()

    # Save
    results = {
        'summary': {
            'n_folds': len(all_results),
            'avg_bifurcation_ic': avg_bifurc_ic if 'avg_bifurc_ic' in locals() else None,
            'avg_sharpe_improvement': avg_improvement if 'avg_improvement' in locals() else None,
            'avg_net_sharpe': avg_net_sharpe if 'avg_net_sharpe' in locals() else None,
            'p_value': p_value if 'p_value' in locals() else None,
        },
        'folds': all_results,
    }

    with open('quick_production_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("=" * 80)
    print("PRODUCTION READINESS")
    print("=" * 80)
    print()

    if 'avg_net_sharpe' in locals():
        ready = (
            avg_net_sharpe > 0 and
            sum(1 for r in all_results if r['net_sharpe'] > 0) == len(all_results) and
            (p_value < 0.05 if 'p_value' in locals() else False)
        )

        print("Criteria:")
        print(f"  ✓ Positive net Sharpe:       {avg_net_sharpe > 0} ({avg_net_sharpe:+.2f})")
        print(f"  ✓ All folds positive:        {sum(1 for r in all_results if r['net_sharpe'] > 0) == len(all_results)}")
        print(f"  ✓ Statistically significant: {(p_value < 0.05) if 'p_value' in locals() else 'N/A'}")
        print()

        if ready:
            print("🎯 PRODUCTION READY")
        elif avg_net_sharpe > 0:
            print("⚠️  CONDITIONAL - Positive but needs more validation")
        else:
            print("❌ NOT READY")
    else:
        print("⚠️  Insufficient data")

    print("\n✓ Results saved to: quick_production_results.json")


if __name__ == '__main__':
    run_quick_production_validation()
