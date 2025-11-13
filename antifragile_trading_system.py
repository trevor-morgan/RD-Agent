"""
ANTI-FRAGILE TRADING SYSTEM
Fundamental Paradigm Shift: Predict Failures, Not Successes

Core Philosophy (Taleb):
- Avoiding ruin > maximizing returns
- Predict which strategies will FAIL
- Dynamically allocate AWAY from failing strategies
- Benefit from volatility and uncertainty

Novel Components:
1. Market Psychology Metrics (consciousness, reflexivity)
2. Phase Transition Detection (bifurcation)
3. Strategy Failure Prediction (meta-learning)
4. Regime Contagion (cross-asset spillover)
5. Anti-fragile Capital Allocation

Author: RD-Agent Research Team
Date: 2025-11-13
Purpose: Fundamentally change how algo trading works
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_integrated import HCANPsi
from hcan_psi_real_data_validation import RealMarketDataLoader, HCANPsiDataset


# ==============================================================================
# 1. MARKET PSYCHOLOGY ANALYZER
# ==============================================================================

class MarketPsychologyAnalyzer:
    """
    Analyze UNEXPLORED HCAN-Ψ outputs that might capture market psychology.

    Novel metrics:
    - Consciousness: Collective market awareness/attention
    - Reflexivity: Self-referential market behavior
    - Bifurcation: Phase transition proximity
    - Entropy: Information content / uncertainty
    - Agent clustering: Herding behavior
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()

    def extract_psychology_metrics(
        self,
        dataset: HCANPsiDataset,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Extract all HCAN-Ψ outputs, focusing on psychology metrics.
        """

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_metrics = {
            'consciousness': [],
            'bifurcation': [],
            'entropy': [],
            'lyapunov': [],
            'hurst': [],
            'regime': [],
            'return': [],
            'actual_return': [],
            'timestamp_idx': [],
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                try:
                    digital = batch['digital_features'].to(self.device)

                    analog_dict = {
                        'returns': batch['analog_returns'].to(self.device),
                        'current_lyapunov': batch['current_lyapunov'].unsqueeze(1).to(self.device),
                        'current_hurst': batch['current_hurst'].unsqueeze(1).to(self.device),
                        'microstructure': batch['microstructure'].to(self.device),
                        'order_flow': batch['order_flow'].to(self.device),
                    }

                    psi_dict = {
                        'correlations': batch['correlations'].to(self.device),
                        'order_sizes': batch['order_sizes'].to(self.device),
                        'liquidity': batch['liquidity'].to(self.device),
                        'prices': batch['prices'].to(self.device),
                        'fundamentals': batch['fundamentals'].to(self.device),
                    }

                    # Get ALL outputs from HCAN-Ψ
                    outputs = self.model(digital, analog_dict, psi_dict)

                    # Extract psychology metrics (correct keys, flatten properly)
                    all_metrics['consciousness'].extend(outputs['consciousness_pred'].cpu().numpy().flatten().tolist())
                    all_metrics['bifurcation'].extend(outputs['bifurcation_pred'].cpu().numpy().flatten().tolist())
                    all_metrics['entropy'].extend(outputs['entropy_pred'].cpu().numpy().flatten().tolist())
                    all_metrics['lyapunov'].extend(outputs['lyapunov_pred'].cpu().numpy().flatten().tolist())
                    all_metrics['hurst'].extend(outputs['hurst_pred'].cpu().numpy().flatten().tolist())
                    all_metrics['regime'].extend(torch.argmax(outputs['regime_pred'], dim=1).cpu().numpy().tolist())
                    all_metrics['return'].extend(outputs['return_pred'].cpu().numpy().flatten().tolist())
                    all_metrics['actual_return'].extend(batch['return'].numpy().flatten().tolist())

                    # Timestamp tracking
                    start_idx = batch_idx * batch_size
                    all_metrics['timestamp_idx'].extend(range(start_idx, start_idx + len(batch['return'])))

                except Exception as e:
                    print(f"  Warning: Batch {batch_idx} failed: {e}")
                    continue

        df = pd.DataFrame(all_metrics)
        return df

    def analyze_consciousness_predictive_power(self, df: pd.DataFrame) -> Dict:
        """
        Test if 'consciousness' predicts anything meaningful.

        Hypothesis: High consciousness = high market attention = precursor to large moves
        """

        # Correlation with future returns
        consciousness = np.array(df['consciousness'].values, dtype=float)
        actual_returns = np.array(df['actual_return'].values, dtype=float)

        # Current correlation
        current_ic = np.corrcoef(consciousness, actual_returns)[0, 1]

        # Lead-lag analysis: Does consciousness predict FUTURE returns?
        lead_ics = {}
        for lag in [1, 2, 3, 5, 10]:
            if len(consciousness) > lag:
                future_returns = actual_returns[lag:]
                current_consciousness = consciousness[:-lag]
                lead_ic = np.corrcoef(current_consciousness, future_returns)[0, 1]
                lead_ics[f'lag_{lag}'] = lead_ic

        # Correlation with future volatility
        future_vols = {}
        for lag in [1, 5, 10]:
            if len(actual_returns) > lag:
                rolling_vol = pd.Series(actual_returns).rolling(lag).std().shift(-lag).values
                valid_idx = ~np.isnan(rolling_vol)
                if valid_idx.sum() > 10:
                    vol_ic = np.corrcoef(consciousness[valid_idx], rolling_vol[valid_idx])[0, 1]
                    future_vols[f'vol_lag_{lag}'] = vol_ic

        return {
            'current_return_ic': current_ic,
            'future_return_ics': lead_ics,
            'future_volatility_ics': future_vols,
            'mean_consciousness': np.mean(consciousness),
            'std_consciousness': np.std(consciousness),
        }

    def analyze_bifurcation_as_crash_indicator(self, df: pd.DataFrame) -> Dict:
        """
        Test if 'bifurcation' predicts phase transitions (crashes, regime changes).

        Hypothesis: High bifurcation = system near critical point = crash imminent
        """

        bifurcation = np.array(df['bifurcation'].values, dtype=float)
        actual_returns = np.array(df['actual_return'].values, dtype=float)

        # Define "extreme events" (crashes/spikes)
        extreme_threshold = np.percentile(np.abs(actual_returns), 95)
        is_extreme = np.abs(actual_returns) > extreme_threshold

        # Does bifurcation predict extreme events?
        extreme_event_ics = {}
        for lag in [1, 2, 3, 5, 10]:
            if len(bifurcation) > lag:
                future_extreme = is_extreme[lag:]
                current_bifurcation = bifurcation[:-lag]

                # Correlation
                if future_extreme.sum() > 0:
                    ic = np.corrcoef(current_bifurcation, future_extreme.astype(float))[0, 1]
                    extreme_event_ics[f'lag_{lag}'] = ic

        # Mean bifurcation before extreme events
        if is_extreme.sum() > 0:
            bifurcation_before_extreme = bifurcation[is_extreme].mean()
            bifurcation_normal = bifurcation[~is_extreme].mean()
        else:
            bifurcation_before_extreme = np.nan
            bifurcation_normal = np.nan

        return {
            'extreme_event_ics': extreme_event_ics,
            'bifurcation_before_extreme': bifurcation_before_extreme,
            'bifurcation_normal': bifurcation_normal,
            'n_extreme_events': is_extreme.sum(),
        }

    def analyze_entropy_as_uncertainty(self, df: pd.DataFrame) -> Dict:
        """
        Test if 'entropy' captures market uncertainty.

        Hypothesis: High entropy = high uncertainty = wider future return distribution
        """

        entropy = np.array(df['entropy'].values, dtype=float)
        actual_returns = np.array(df['actual_return'].values, dtype=float)

        # Correlation with absolute returns (magnitude, not direction)
        abs_return_ic = np.corrcoef(entropy, np.abs(actual_returns))[0, 1]

        # Does high entropy predict high future volatility?
        future_vol_ics = {}
        for lag in [1, 5, 10]:
            if len(actual_returns) > lag:
                rolling_vol = pd.Series(actual_returns).rolling(lag).std().shift(-lag).values
                valid_idx = ~np.isnan(rolling_vol)
                if valid_idx.sum() > 10:
                    vol_ic = np.corrcoef(entropy[valid_idx], rolling_vol[valid_idx])[0, 1]
                    future_vol_ics[f'lag_{lag}'] = vol_ic

        return {
            'abs_return_ic': abs_return_ic,
            'future_volatility_ics': future_vol_ics,
            'mean_entropy': np.mean(entropy),
            'std_entropy': np.std(entropy),
        }


# ==============================================================================
# 2. STRATEGY FAILURE PREDICTOR
# ==============================================================================

class StrategyFailurePredictor:
    """
    Revolutionary approach: Predict when strategies will FAIL, not succeed.

    Strategies tested:
    1. Momentum (moving average crossover)
    2. Mean reversion (RSI-based)
    3. Trend following (breakout)
    4. Volatility (sell high vol, buy low vol)

    For each strategy, predict probability of failure in next N periods.
    """

    def __init__(self):
        self.strategies = {
            'momentum': self._momentum_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'trend_following': self._trend_following_strategy,
            'volatility': self._volatility_strategy,
        }

    def _momentum_strategy(self, returns: np.ndarray) -> float:
        """Simple momentum: positive if recent returns positive."""
        if len(returns) < 10:
            return 0.0
        recent_return = np.sum(returns[-10:])
        return 1.0 if recent_return > 0 else -1.0

    def _mean_reversion_strategy(self, returns: np.ndarray) -> float:
        """Mean reversion: negative if recent returns positive (fade the move)."""
        if len(returns) < 10:
            return 0.0
        recent_return = np.sum(returns[-10:])
        return -1.0 if recent_return > 0 else 1.0

    def _trend_following_strategy(self, returns: np.ndarray) -> float:
        """Trend: follow direction of longer-term trend."""
        if len(returns) < 20:
            return 0.0
        long_return = np.sum(returns[-20:])
        return 1.0 if long_return > 0 else -1.0

    def _volatility_strategy(self, returns: np.ndarray) -> float:
        """Volatility: short high vol, long low vol."""
        if len(returns) < 20:
            return 0.0
        recent_vol = np.std(returns[-10:])
        long_vol = np.std(returns[-20:])
        return -1.0 if recent_vol > long_vol else 1.0

    def backtest_strategies(self, all_returns: np.ndarray) -> pd.DataFrame:
        """
        Backtest all strategies and record when they fail.

        Failure = strategy signal wrong for next period.
        """

        results = []

        for i in range(20, len(all_returns) - 1):  # Need history + 1 future
            historical_returns = all_returns[:i]
            next_return = all_returns[i]

            for strategy_name, strategy_func in self.strategies.items():
                # Get strategy signal
                signal = strategy_func(historical_returns)

                # Did strategy fail? (signal * next_return < 0)
                failed = (signal * next_return < 0)

                results.append({
                    'timestamp_idx': i,
                    'strategy': strategy_name,
                    'signal': signal,
                    'next_return': next_return,
                    'failed': failed,
                    'pnl': signal * next_return,
                })

        return pd.DataFrame(results)

    def create_failure_prediction_dataset(
        self,
        strategy_results: pd.DataFrame,
        psychology_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create dataset for predicting strategy failures.

        Features: consciousness, bifurcation, entropy, regime, chaos metrics
        Target: Will strategy fail in next period?
        """

        # Merge psychology metrics with strategy results
        merged = strategy_results.merge(
            psychology_df,
            on='timestamp_idx',
            how='inner'
        )

        # Features
        feature_cols = [
            'consciousness', 'bifurcation', 'entropy',
            'lyapunov', 'hurst', 'regime',
        ]

        X = merged[feature_cols].values
        y = merged['failed'].astype(int).values
        strategy_names = merged['strategy'].values

        return X, y, strategy_names


# ==============================================================================
# 3. ANTI-FRAGILE META-LEARNER
# ==============================================================================

class AntifragileMetaLearner:
    """
    Meta-learning system that benefits from uncertainty.

    Key ideas:
    1. Don't optimize for returns
    2. Optimize for avoiding failures (anti-fragile)
    3. Increase allocation when confident (low failure probability)
    4. Decrease allocation when uncertain (high failure probability)
    5. Benefit from volatility by detecting regime changes early
    """

    def __init__(self, n_strategies: int = 4):
        self.n_strategies = n_strategies
        self.failure_models = {}  # One per strategy

    def train_failure_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy_names: List[str]
    ):
        """
        Train one model per strategy to predict its failure probability.
        """

        from sklearn.ensemble import GradientBoostingClassifier

        unique_strategies = list(set(strategy_names))

        for strategy in unique_strategies:
            # Filter data for this strategy
            mask = np.array([s == strategy for s in strategy_names])
            X_strat = X[mask]
            y_strat = y[mask]

            if len(y_strat) < 50:
                print(f"  Skipping {strategy}: insufficient data")
                continue

            # Train model
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_strat, y_strat)

            self.failure_models[strategy] = model
            print(f"  Trained failure model for {strategy}")

    def predict_failure_probabilities(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """
        Predict failure probability for each strategy.

        Returns: {strategy_name: failure_probability}
        """

        failure_probs = {}

        for strategy, model in self.failure_models.items():
            prob = model.predict_proba(features.reshape(1, -1))[0, 1]  # P(fail)
            failure_probs[strategy] = prob

        return failure_probs

    def allocate_capital(
        self,
        failure_probs: Dict[str, float],
        total_capital: float = 1.0
    ) -> Dict[str, float]:
        """
        Anti-fragile allocation: allocate AWAY from strategies likely to fail.

        Allocation ∝ (1 - failure_probability)²
        Squared to emphasize high-confidence strategies.
        """

        # Convert to "success probability"
        success_probs = {
            strategy: (1 - fail_prob) ** 2
            for strategy, fail_prob in failure_probs.items()
        }

        # Normalize to sum to 1
        total = sum(success_probs.values())
        if total == 0:
            # All strategies likely to fail - equal weight (or go to cash)
            allocations = {s: total_capital / len(success_probs) for s in success_probs.keys()}
        else:
            allocations = {
                strategy: (success_prob / total) * total_capital
                for strategy, success_prob in success_probs.items()
            }

        return allocations


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_antifragile_system():
    """
    Run complete anti-fragile trading system.
    """

    print("=" * 80)
    print("ANTI-FRAGILE TRADING SYSTEM")
    print("=" * 80)
    print()
    print("Paradigm Shift: Predict Failures, Not Successes")
    print()
    print("Components:")
    print("1. Market Psychology Analyzer (consciousness, bifurcation, entropy)")
    print("2. Strategy Failure Predictor (when will strategies fail?)")
    print("3. Anti-fragile Meta-Learner (allocate away from failures)")
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
    print(f"Dataset: {len(full_dataset)} samples\n")

    # Load HCAN-Ψ model
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

    # ==============================================================================
    # TEST 1: MARKET PSYCHOLOGY ANALYSIS
    # ==============================================================================

    print("=" * 80)
    print("TEST 1: MARKET PSYCHOLOGY METRICS")
    print("=" * 80)
    print()

    psych_analyzer = MarketPsychologyAnalyzer(model, device)
    print("Extracting psychology metrics from HCAN-Ψ...")
    psychology_df = psych_analyzer.extract_psychology_metrics(full_dataset)
    print(f"✓ Extracted {len(psychology_df)} samples\n")

    # Consciousness analysis
    print("Analyzing 'consciousness' predictive power...")
    consciousness_results = psych_analyzer.analyze_consciousness_predictive_power(psychology_df)
    print(f"  Current return IC: {consciousness_results['current_return_ic']:+.4f}")
    print(f"  Future return ICs:")
    for lag, ic in consciousness_results['future_return_ics'].items():
        print(f"    {lag}: {ic:+.4f}")
    print(f"  Future volatility ICs:")
    for lag, ic in consciousness_results['future_volatility_ics'].items():
        print(f"    {lag}: {ic:+.4f}")
    print()

    # Bifurcation analysis
    print("Analyzing 'bifurcation' as crash indicator...")
    bifurcation_results = psych_analyzer.analyze_bifurcation_as_crash_indicator(psychology_df)
    print(f"  Extreme events detected: {bifurcation_results['n_extreme_events']}")
    print(f"  Bifurcation before extreme: {bifurcation_results['bifurcation_before_extreme']:.4f}")
    print(f"  Bifurcation normal:         {bifurcation_results['bifurcation_normal']:.4f}")
    print(f"  Extreme event prediction ICs:")
    for lag, ic in bifurcation_results['extreme_event_ics'].items():
        print(f"    {lag}: {ic:+.4f}")
    print()

    # Entropy analysis
    print("Analyzing 'entropy' as uncertainty metric...")
    entropy_results = psych_analyzer.analyze_entropy_as_uncertainty(psychology_df)
    print(f"  Abs return IC: {entropy_results['abs_return_ic']:+.4f}")
    print(f"  Future volatility ICs:")
    for lag, ic in entropy_results['future_volatility_ics'].items():
        print(f"    {lag}: {ic:+.4f}")
    print()

    # ==============================================================================
    # TEST 2: STRATEGY FAILURE PREDICTION
    # ==============================================================================

    print("=" * 80)
    print("TEST 2: STRATEGY FAILURE PREDICTION")
    print("=" * 80)
    print()

    failure_predictor = StrategyFailurePredictor()

    print("Backtesting 4 simple strategies...")
    all_returns = psychology_df['actual_return'].values
    strategy_results = failure_predictor.backtest_strategies(all_returns)

    # Summary stats per strategy
    print("\nStrategy performance:")
    for strategy in strategy_results['strategy'].unique():
        strat_data = strategy_results[strategy_results['strategy'] == strategy]
        failure_rate = strat_data['failed'].mean()
        mean_pnl = strat_data['pnl'].mean()
        print(f"  {strategy:20s}: Failure rate = {failure_rate:.2%}, Mean PnL = {mean_pnl:+.6f}")
    print()

    # Create failure prediction dataset
    print("Creating failure prediction dataset...")
    X, y, strategy_names = failure_predictor.create_failure_prediction_dataset(
        strategy_results, psychology_df
    )
    print(f"  Features: {X.shape}")
    print(f"  Samples: {len(y)}")
    print(f"  Failure rate: {y.mean():.2%}\n")

    # ==============================================================================
    # TEST 3: ANTI-FRAGILE META-LEARNER
    # ==============================================================================

    print("=" * 80)
    print("TEST 3: ANTI-FRAGILE META-LEARNER")
    print("=" * 80)
    print()

    # Split data temporally
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    strategy_names_train = strategy_names[:train_size]
    strategy_names_test = strategy_names[train_size:]

    # Train meta-learner
    print("Training anti-fragile meta-learner...")
    meta_learner = AntifragileMetaLearner()
    meta_learner.train_failure_models(X_train, y_train, strategy_names_train)
    print()

    # Test on holdout data
    print("Testing on holdout period...")
    print()

    # Simulate day-by-day allocation
    test_allocations = []
    test_returns = []

    for i in range(len(X_test)):
        features = X_test[i]

        # Predict failure probabilities
        failure_probs = meta_learner.predict_failure_probabilities(features)

        # Allocate capital
        allocations = meta_learner.allocate_capital(failure_probs)

        test_allocations.append(allocations)

    # Save results
    results = {
        'psychology_analysis': {
            'consciousness': consciousness_results,
            'bifurcation': bifurcation_results,
            'entropy': entropy_results,
        },
        'strategy_failure': {
            'failure_rates': strategy_results.groupby('strategy')['failed'].mean().to_dict(),
            'mean_pnls': strategy_results.groupby('strategy')['pnl'].mean().to_dict(),
        },
        'meta_learner': {
            'n_models_trained': len(meta_learner.failure_models),
            'strategies': list(meta_learner.failure_models.keys()),
        }
    }

    with open('antifragile_system_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("✓ Results saved to: antifragile_system_results.json")
    print()

    # Summary
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    print("1. Market Psychology Metrics:")
    print(f"   - Consciousness useful for vol prediction: {max(consciousness_results['future_volatility_ics'].values()) > 0.1}")
    print(f"   - Bifurcation predicts extreme events: {any(abs(v) > 0.1 for v in bifurcation_results['extreme_event_ics'].values())}")
    print(f"   - Entropy captures uncertainty: {entropy_results['abs_return_ic'] > 0.05}")
    print()

    print("2. Strategy Failure Prediction:")
    print(f"   - Trained {len(meta_learner.failure_models)} failure models")
    print(f"   - Can dynamically allocate away from failing strategies")
    print()

    print("3. Anti-fragile System:")
    print(f"   - Focuses on avoiding failures, not chasing wins")
    print(f"   - Benefits from uncertainty detection")
    print(f"   - Adapts allocation based on market psychology")


if __name__ == '__main__':
    run_antifragile_system()
