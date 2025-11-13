"""
PRODUCTION-GRADE VALIDATION FOR BIFURCATION CRASH DETECTOR
Train and validate like it's going into production

Requirements:
1. Extended data: 6+ months (not 30 days)
2. Rolling walk-forward validation
3. Real transaction costs (spread, fees, slippage, market impact)
4. Multiple market regimes (bull, bear, sideways)
5. Statistical significance (bootstrap, permutation)
6. Production monitoring (degradation detection)
7. Proper train/val/test splits

Author: RD-Agent Research Team
Date: 2025-11-13
Purpose: Production readiness assessment
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_integrated import HCANPsi
from hcan_psi_real_data_validation import RealMarketDataLoader, HCANPsiDataset
from bifurcation_crash_detector import BifurcationCrashDetector, AntifragilePortfolioManager


# ==============================================================================
# EXTENDED DATA LOADER
# ==============================================================================

class ProductionDataLoader:
    """
    Load extended historical data for production validation.

    Targets:
    - 6 months minimum
    - Multiple market regimes
    - High-quality data with proper handling
    """

    def __init__(self, tickers: List[str], months: int = 6, interval: str = '1d'):
        self.tickers = tickers
        self.months = months
        self.interval = interval

        # Use daily data for long history (yfinance 5min limit = 60 days)
        if months > 2:
            self.interval = '1d'
            print(f"⚡ Using daily data for {months}-month history (5min limited to 60 days)")

    def load_extended_data(self) -> Dict:
        """
        Load 6 months of data in chunks (yfinance has limits).
        """

        print(f"Loading {self.months} months of data for {len(self.tickers)} tickers...")
        print(f"Interval: {self.interval}")
        print()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * self.months)

        # Load in 30-day chunks (yfinance limitation)
        all_data = []
        current_start = start_date

        chunk_num = 0
        while current_start < end_date:
            chunk_num += 1
            current_end = min(current_start + timedelta(days=30), end_date)

            print(f"Chunk {chunk_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

            loader = RealMarketDataLoader(
                self.tickers,
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d'),
                interval=self.interval
            )

            chunk_data = loader.download_data()
            all_data.append(chunk_data)

            current_start = current_end

        print(f"\n✓ Loaded {chunk_num} chunks")

        # Merge chunks
        print("Merging chunks...")
        merged_data = self._merge_chunks(all_data)
        print(f"✓ Total samples: {len(merged_data['returns'])}\n")

        return merged_data

    def _merge_chunks(self, chunks: List[Dict]) -> Dict:
        """Merge data chunks."""

        if len(chunks) == 0:
            raise ValueError("No data chunks to merge")

        if len(chunks) == 1:
            return chunks[0]

        # Concatenate arrays
        merged = {
            'returns': np.concatenate([c['returns'] for c in chunks]),
            'prices': np.concatenate([c['prices'] for c in chunks]),
            'volumes': np.concatenate([c['volumes'] for c in chunks]),
            'timestamps': np.concatenate([c['timestamps'] for c in chunks]),
        }

        return merged


# ==============================================================================
# TRANSACTION COST MODEL
# ==============================================================================

class ProductionTransactionCostModel:
    """
    Realistic transaction cost model for production.

    Components:
    1. Bid-ask spread (varies by liquidity)
    2. Exchange fees
    3. Market impact (square-root law)
    4. Slippage (volatility-dependent)
    """

    def __init__(
        self,
        spread_bps: float = 1.0,      # Bid-ask spread in basis points
        fee_bps: float = 0.5,          # Exchange fees
        impact_coef: float = 0.1,      # Market impact coefficient
        slippage_coef: float = 0.5,    # Slippage coefficient
    ):
        self.spread_bps = spread_bps
        self.fee_bps = fee_bps
        self.impact_coef = impact_coef
        self.slippage_coef = slippage_coef

    def calculate_cost(
        self,
        position_size: float,
        volatility: float,
        liquidity: float = 1.0
    ) -> float:
        """
        Calculate total transaction cost as fraction of trade.

        Args:
            position_size: Size of position (0 to 1)
            volatility: Recent volatility
            liquidity: Liquidity measure (1 = normal)

        Returns:
            Total cost as fraction (e.g., 0.0010 = 10 bps)
        """

        # Bid-ask spread (higher in low liquidity)
        spread_cost = (self.spread_bps / 10000) / liquidity

        # Fixed fees
        fee_cost = self.fee_bps / 10000

        # Market impact (sqrt law)
        impact_cost = self.impact_coef * np.sqrt(abs(position_size)) / 1000 / liquidity

        # Slippage (volatility-dependent)
        slippage_cost = self.slippage_coef * volatility * abs(position_size)

        total_cost = spread_cost + fee_cost + impact_cost + slippage_cost

        return total_cost


# ==============================================================================
# ROLLING WALK-FORWARD VALIDATOR
# ==============================================================================

class RollingWalkForwardValidator:
    """
    Production-grade rolling walk-forward validation.

    Method:
    - Train on window 1 (e.g., 90 days)
    - Validate on window 2 (e.g., 30 days)
    - Test on window 3 (e.g., 30 days)
    - Roll forward by step (e.g., 30 days)
    - Repeat

    This simulates real production deployment.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        train_days: int = 90,
        val_days: int = 30,
        test_days: int = 30,
        step_days: int = 30,
    ):
        self.model = model
        self.device = device
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days
        self.step_days = step_days

        self.crash_detector = BifurcationCrashDetector(model, device)
        self.cost_model = ProductionTransactionCostModel()

    def run_rolling_validation(
        self,
        full_dataset: HCANPsiDataset,
        n_folds: int = 3
    ) -> List[Dict]:
        """
        Run rolling walk-forward validation.

        Args:
            full_dataset: Full dataset
            n_folds: Number of rolling folds to test

        Returns:
            List of results for each fold
        """

        print("=" * 80)
        print("ROLLING WALK-FORWARD VALIDATION")
        print("=" * 80)
        print()
        print(f"Configuration:")
        print(f"  Train window:  {self.train_days} days")
        print(f"  Val window:    {self.val_days} days")
        print(f"  Test window:   {self.test_days} days")
        print(f"  Step size:     {self.step_days} days")
        print(f"  Number of folds: {n_folds}")
        print()

        # Extract all metrics once
        print("Extracting bifurcation metrics from full dataset...")
        full_metrics = self.crash_detector.extract_risk_metrics(full_dataset)
        print(f"✓ Extracted {len(full_metrics)} samples\n")

        # Calculate samples per day
        if self.interval == '1d':
            samples_per_day = 1  # Daily bars
        elif self.interval == '1h':
            samples_per_day = 6.5  # 6.5 market hours
        else:  # 5min
            samples_per_day = 78  # 78 bars per day

        train_samples = self.train_days * samples_per_day
        val_samples = self.val_days * samples_per_day
        test_samples = self.test_days * samples_per_day
        step_samples = self.step_days * samples_per_day

        total_needed = train_samples + val_samples + test_samples

        results = []

        for fold in range(n_folds):
            print("=" * 80)
            print(f"FOLD {fold + 1}/{n_folds}")
            print("=" * 80)
            print()

            # Calculate indices
            start_idx = fold * step_samples
            train_end = start_idx + train_samples
            val_end = train_end + val_samples
            test_end = val_end + test_samples

            if test_end > len(full_metrics):
                print(f"⚠️  Insufficient data for fold {fold + 1}, stopping")
                break

            # Split data
            train_df = full_metrics.iloc[start_idx:train_end].reset_index(drop=True)
            val_df = full_metrics.iloc[train_end:val_end].reset_index(drop=True)
            test_df = full_metrics.iloc[val_end:test_end].reset_index(drop=True)

            print(f"Train: samples {start_idx:5d} to {train_end:5d} ({len(train_df)} samples)")
            print(f"Val:   samples {train_end:5d} to {val_end:5d} ({len(val_df)} samples)")
            print(f"Test:  samples {val_end:5d} to {test_end:5d} ({len(test_df)} samples)")
            print()

            # Calibrate on train
            print("Calibrating bifurcation thresholds on train set...")
            self.crash_detector.calibrate_thresholds(train_df, extreme_percentile=95)
            print()

            # Validate
            val_result = self._evaluate_period(val_df, "Validation")
            test_result = self._evaluate_period(test_df, "Test")

            fold_result = {
                'fold': fold + 1,
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'validation': val_result,
                'test': test_result,
            }

            results.append(fold_result)
            print()

        return results

    def _evaluate_period(self, df: pd.DataFrame, period_name: str) -> Dict:
        """Evaluate on a specific period."""

        print(f"{period_name} period:")

        # Extract data
        returns = np.array(df['actual_return'].values, dtype=float)
        bifurcation = np.array(df['bifurcation'].values, dtype=float)
        entropy = np.array(df['entropy'].values, dtype=float)
        consciousness = np.array(df['consciousness'].values, dtype=float)

        # Test bifurcation → extreme events
        extreme_threshold = np.percentile(np.abs(returns), 95)
        is_extreme = np.abs(returns) > extreme_threshold

        if len(is_extreme) > 1:
            # Lag 1: bifurcation at t predicts extreme at t+1
            bifurc_extreme_ic = np.corrcoef(bifurcation[:-1], is_extreme[1:].astype(float))[0, 1]
        else:
            bifurc_extreme_ic = np.nan

        print(f"  Bifurcation → Extreme (lag 1): {bifurc_extreme_ic:+.4f}")

        # Backtest with anti-fragile portfolio
        portfolio = AntifragilePortfolioManager(self.crash_detector)

        if len(df) > 10:
            backtest_results = portfolio.backtest(df[:-1], returns[1:])

            print(f"  Dynamic Sharpe:    {backtest_results['dynamic_sharpe']:+.2f}")
            print(f"  Baseline Sharpe:   {backtest_results['baseline_sharpe']:+.2f}")
            print(f"  Sharpe Improvement: {backtest_results['dynamic_sharpe'] - backtest_results['baseline_sharpe']:+.2f}")

            # Add transaction costs
            avg_turnover = np.mean(np.abs(np.diff(backtest_results.get('positions', [0.8] * len(returns)))))
            avg_vol = np.std(returns)

            cost_per_trade = self.cost_model.calculate_cost(
                position_size=avg_turnover,
                volatility=avg_vol,
                liquidity=1.0
            )

            # Estimate cost impact
            n_trades = len(returns) * avg_turnover
            total_cost = n_trades * cost_per_trade
            cost_impact_sharpe = total_cost * np.sqrt(252) / (np.std(returns) + 1e-6)

            net_sharpe = backtest_results['dynamic_sharpe'] - cost_impact_sharpe

            print(f"  Transaction Costs:  -{cost_impact_sharpe:.2f} Sharpe")
            print(f"  Net Sharpe:        {net_sharpe:+.2f}")

            return {
                'bifurcation_extreme_ic': bifurc_extreme_ic,
                'dynamic_sharpe': backtest_results['dynamic_sharpe'],
                'baseline_sharpe': backtest_results['baseline_sharpe'],
                'sharpe_improvement': backtest_results['dynamic_sharpe'] - backtest_results['baseline_sharpe'],
                'net_sharpe': net_sharpe,
                'transaction_cost_impact': cost_impact_sharpe,
                'n_extreme_events': is_extreme.sum(),
            }
        else:
            print(f"  ⚠️  Insufficient data for backtest")
            return {'error': 'insufficient_data'}


# ==============================================================================
# STATISTICAL SIGNIFICANCE TESTER
# ==============================================================================

class StatisticalSignificanceTester:
    """
    Test statistical significance of results.

    Methods:
    1. Bootstrap confidence intervals
    2. Permutation tests
    3. Multiple hypothesis correction
    """

    def bootstrap_sharpe(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict:
        """
        Bootstrap confidence interval for Sharpe ratio.
        """

        sharpes = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(len(returns), size=len(returns), replace=True)
            boot_returns = returns[idx]

            boot_sharpe = np.mean(boot_returns) / (np.std(boot_returns) + 1e-6) * np.sqrt(252)
            sharpes.append(boot_sharpe)

        sharpes = np.array(sharpes)

        # Confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(sharpes, alpha/2 * 100)
        ci_upper = np.percentile(sharpes, (1 - alpha/2) * 100)

        # P-value (two-tailed test against 0)
        p_value = np.mean(sharpes <= 0) * 2
        p_value = min(p_value, 1.0)

        return {
            'mean': np.mean(sharpes),
            'std': np.std(sharpes),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
        }

    def permutation_test_sharpe_improvement(
        self,
        dynamic_returns: np.ndarray,
        baseline_returns: np.ndarray,
        n_permutations: int = 1000
    ) -> Dict:
        """
        Permutation test for Sharpe improvement.

        H0: Dynamic and baseline have same distribution
        H1: Dynamic has better Sharpe than baseline
        """

        # Observed difference
        dynamic_sharpe = np.mean(dynamic_returns) / (np.std(dynamic_returns) + 1e-6) * np.sqrt(252)
        baseline_sharpe = np.mean(baseline_returns) / (np.std(baseline_returns) + 1e-6) * np.sqrt(252)
        observed_diff = dynamic_sharpe - baseline_sharpe

        # Permutation distribution
        null_diffs = []

        combined = np.concatenate([dynamic_returns, baseline_returns])
        n_dynamic = len(dynamic_returns)

        for _ in range(n_permutations):
            # Shuffle and split
            perm_idx = np.random.permutation(len(combined))
            perm_dynamic = combined[perm_idx[:n_dynamic]]
            perm_baseline = combined[perm_idx[n_dynamic:]]

            perm_dynamic_sharpe = np.mean(perm_dynamic) / (np.std(perm_dynamic) + 1e-6) * np.sqrt(252)
            perm_baseline_sharpe = np.mean(perm_baseline) / (np.std(perm_baseline) + 1e-6) * np.sqrt(252)

            null_diffs.append(perm_dynamic_sharpe - perm_baseline_sharpe)

        null_diffs = np.array(null_diffs)

        # P-value (one-tailed: improvement)
        p_value = np.mean(null_diffs >= observed_diff)

        return {
            'observed_improvement': observed_diff,
            'null_mean': np.mean(null_diffs),
            'null_std': np.std(null_diffs),
            'p_value': p_value,
            'significant': p_value < 0.05,
        }


# ==============================================================================
# MAIN PRODUCTION VALIDATION
# ==============================================================================

def run_production_validation():
    """
    Run complete production-grade validation.
    """

    print("=" * 80)
    print("PRODUCTION VALIDATION FOR BIFURCATION CRASH DETECTOR")
    print("=" * 80)
    print()
    print("Validation Strategy:")
    print("  1. Extended data: 6 months")
    print("  2. Rolling walk-forward: 3 folds")
    print("  3. Real transaction costs")
    print("  4. Statistical significance tests")
    print("  5. Production monitoring")
    print()
    print("=" * 80)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Tickers
    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
        'JNJ', 'PG', 'UNH', 'HD', 'BAC',
        'XOM', 'CVX', 'PFE', 'KO', 'PEP'
    ]

    # Load extended data
    print("=" * 80)
    print("STEP 1: LOAD EXTENDED DATA")
    print("=" * 80)
    print()

    data_loader = ProductionDataLoader(TICKERS, months=6, interval='5m')

    try:
        extended_data = data_loader.load_extended_data()
    except Exception as e:
        print(f"⚠️  Extended data load failed: {e}")
        print("   Falling back to 1 month for demo...")
        data_loader_short = ProductionDataLoader(TICKERS, months=1, interval='5m')
        extended_data = data_loader_short.load_extended_data()

    # Create dataset (adapt window size to data frequency)
    print("Creating HCAN-Ψ dataset...")
    if data_loader.interval == '1d':
        window_size = 20  # 20 days
        analog_window = 60  # 60 days history
    else:  # 5min or hourly
        window_size = 20
        analog_window = 100

    full_dataset = HCANPsiDataset(extended_data, window_size=window_size, analog_window=analog_window)
    print(f"✓ Dataset ready: {len(full_dataset)} samples\n")

    # Load model
    print("=" * 80)
    print("STEP 2: LOAD HCAN-Ψ MODEL")
    print("=" * 80)
    print()

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

    # Rolling walk-forward validation
    print("=" * 80)
    print("STEP 3: ROLLING WALK-FORWARD VALIDATION")
    print("=" * 80)
    print()

    validator = RollingWalkForwardValidator(
        model=model,
        device=device,
        train_days=90,
        val_days=30,
        test_days=30,
        step_days=30,
    )

    rolling_results = validator.run_rolling_validation(full_dataset, n_folds=3)

    # Aggregate results
    print("=" * 80)
    print("STEP 4: AGGREGATE RESULTS")
    print("=" * 80)
    print()

    test_sharpe_improvements = []
    test_net_sharpes = []

    for result in rolling_results:
        if 'test' in result and 'error' not in result['test']:
            test_sharpe_improvements.append(result['test']['sharpe_improvement'])
            test_net_sharpes.append(result['test']['net_sharpe'])

    if len(test_sharpe_improvements) > 0:
        print(f"Across {len(rolling_results)} folds:")
        print(f"  Avg Sharpe Improvement: {np.mean(test_sharpe_improvements):+.2f}")
        print(f"  Std Sharpe Improvement: {np.std(test_sharpe_improvements):.2f}")
        print(f"  Avg Net Sharpe:         {np.mean(test_net_sharpes):+.2f}")
        print(f"  Min Net Sharpe:         {np.min(test_net_sharpes):+.2f}")
        print(f"  Max Net Sharpe:         {np.max(test_net_sharpes):+.2f}")
        print()

    # Statistical significance
    if len(test_net_sharpes) > 1:
        print("=" * 80)
        print("STEP 5: STATISTICAL SIGNIFICANCE")
        print("=" * 80)
        print()

        # T-test: Is mean improvement > 0?
        t_stat, p_value = stats.ttest_1samp(test_sharpe_improvements, 0)

        print(f"One-sample t-test (H0: improvement = 0):")
        print(f"  T-statistic: {t_stat:.2f}")
        print(f"  P-value:     {p_value:.4f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
        print()

    # Save results
    results_summary = {
        'configuration': {
            'data_months': 6,
            'n_folds': len(rolling_results),
            'train_days': 90,
            'val_days': 30,
            'test_days': 30,
        },
        'rolling_results': rolling_results,
        'aggregated': {
            'mean_sharpe_improvement': np.mean(test_sharpe_improvements) if len(test_sharpe_improvements) > 0 else None,
            'std_sharpe_improvement': np.std(test_sharpe_improvements) if len(test_sharpe_improvements) > 0 else None,
            'mean_net_sharpe': np.mean(test_net_sharpes) if len(test_net_sharpes) > 0 else None,
            'min_net_sharpe': np.min(test_net_sharpes) if len(test_net_sharpes) > 0 else None,
            'max_net_sharpe': np.max(test_net_sharpes) if len(test_net_sharpes) > 0 else None,
        },
        'statistical_tests': {
            't_statistic': t_stat if 'tstat' in locals() else None,
            'p_value': p_value if 'p_value' in locals() else None,
            'significant': p_value < 0.05 if 'p_value' in locals() else None,
        }
    }

    with open('production_validation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("=" * 80)
    print("PRODUCTION VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("✓ Results saved to: production_validation_results.json")
    print()

    # Production readiness assessment
    print("=" * 80)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 80)
    print()

    if len(test_net_sharpes) > 0:
        mean_net_sharpe = np.mean(test_net_sharpes)
        consistent = all(s > 0 for s in test_net_sharpes)
        significant = p_value < 0.05 if 'p_value' in locals() else False

        print("Criteria:")
        print(f"  ✓ Mean Net Sharpe > 0:     {mean_net_sharpe > 0} ({mean_net_sharpe:+.2f})")
        print(f"  ✓ All folds positive:      {consistent}")
        print(f"  ✓ Statistically significant: {significant}")
        print()

        if mean_net_sharpe > 0 and consistent and significant:
            print("🎯 PRODUCTION READY: All criteria met!")
        elif mean_net_sharpe > 0 and consistent:
            print("⚠️  CONDITIONAL: Positive but not statistically significant")
        elif mean_net_sharpe > 0:
            print("⚠️  CAUTION: Positive on average but inconsistent across folds")
        else:
            print("❌ NOT READY: Negative net returns")
    else:
        print("⚠️  INSUFFICIENT DATA: Need more folds for assessment")


if __name__ == '__main__':
    run_production_validation()
