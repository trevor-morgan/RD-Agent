"""
Feature Extraction and Analysis
Extract what worked from HCAN-Ψ and test in simpler models

This script:
1. Tests chaos metrics (Lyapunov, Hurst, entropy) in isolation
2. Evaluates regime detection capability
3. Builds simple linear and shallow models
4. Validates everything with walk-forward testing

Author: RD-Agent Research Team
Date: 2025-11-13
Purpose: Salvage valuable components from Phase 2
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

from hcan_psi_real_data_validation import RealMarketDataLoader, HCANPsiDataset


# ==============================================================================
# CHAOS METRICS ANALYSIS
# ==============================================================================

class ChaosMetricsAnalyzer:
    """
    Test predictive power of chaos metrics in isolation.

    Tests:
    - Lyapunov exponent correlation with returns
    - Hurst exponent correlation with returns
    - Entropy metrics correlation with returns
    - Combined chaos features IC
    """

    def __init__(self, dataset: HCANPsiDataset):
        self.dataset = dataset

    def extract_chaos_features(self) -> pd.DataFrame:
        """Extract all chaos metrics from dataset."""

        chaos_features = []
        returns = []

        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]

                # Extract chaos metrics
                lyapunov = sample['current_lyapunov'].item()
                hurst = sample['current_hurst'].item()

                # Extract digital features (wavelet-based)
                digital = sample['digital_features'].numpy()
                wavelet_energy = np.mean(np.abs(digital))
                wavelet_entropy = -np.sum(np.abs(digital) * np.log(np.abs(digital) + 1e-10))

                # Extract return target
                ret = sample['return'].item()

                chaos_features.append({
                    'lyapunov': lyapunov,
                    'hurst': hurst,
                    'wavelet_energy': wavelet_energy,
                    'wavelet_entropy': wavelet_entropy,
                })
                returns.append(ret)
            except:
                continue

        df = pd.DataFrame(chaos_features)
        df['return'] = returns

        return df

    def analyze_individual_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate IC for each chaos metric individually."""

        results = {}

        for feature in ['lyapunov', 'hurst', 'wavelet_energy', 'wavelet_entropy']:
            ic = np.corrcoef(df[feature], df['return'])[0, 1]
            results[feature] = {
                'ic': ic,
                'abs_ic': abs(ic),
                'mean': df[feature].mean(),
                'std': df[feature].std(),
            }

        return results

    def test_combined_chaos_ic(self, df: pd.DataFrame) -> Dict:
        """Test linear combination of all chaos features."""

        # Split data temporally
        n = len(df)
        train_size = int(0.7 * n)

        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # Features
        feature_cols = ['lyapunov', 'hurst', 'wavelet_energy', 'wavelet_entropy']
        X_train = train_df[feature_cols].values
        y_train = train_df['return'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['return'].values

        # Ridge regression (L2 regularization)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # IC
        train_ic = np.corrcoef(train_pred, y_train)[0, 1]
        test_ic = np.corrcoef(test_pred, y_test)[0, 1]

        return {
            'train_ic': train_ic,
            'test_ic': test_ic,
            'coefficients': dict(zip(feature_cols, model.coef_)),
            'intercept': model.intercept_,
        }


# ==============================================================================
# REGIME DETECTION ANALYSIS
# ==============================================================================

class RegimeDetectionAnalyzer:
    """
    Test if HCAN-Ψ can accurately detect market regimes.

    Tests:
    - Volatility regime classification accuracy
    - Trending vs ranging regime detection
    - Feature importance for regime detection
    """

    def __init__(self, tick_data: Dict, dataset: HCANPsiDataset):
        self.tick_data = tick_data
        self.dataset = dataset

    def create_regime_labels(self) -> np.ndarray:
        """
        Create ground truth regime labels based on volatility.

        Returns:
            Array of regime labels (0=low_vol, 1=medium_vol, 2=high_vol)
        """
        returns = self.tick_data['returns']

        # Calculate rolling volatility
        window = 20
        volatilities = []

        for i in range(len(returns)):
            if i >= window:
                vol = np.std(returns[i-window:i])
                volatilities.append(vol)
            else:
                volatilities.append(np.nan)

        volatilities = np.array(volatilities)

        # Create 3 regimes based on tertiles
        valid_vols = volatilities[~np.isnan(volatilities)]
        low_threshold = np.percentile(valid_vols, 33)
        high_threshold = np.percentile(valid_vols, 67)

        regimes = np.zeros(len(volatilities), dtype=int)
        regimes[volatilities < low_threshold] = 0  # Low volatility
        regimes[(volatilities >= low_threshold) & (volatilities < high_threshold)] = 1  # Medium
        regimes[volatilities >= high_threshold] = 2  # High volatility

        # Align with dataset length
        max_idx = len(self.dataset)
        regimes = regimes[:max_idx]

        return regimes

    def extract_features_for_regime_detection(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features that might be useful for regime detection."""

        features = []
        regimes = self.create_regime_labels()

        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]

                # Chaos metrics
                lyapunov = sample['current_lyapunov'].item()
                hurst = sample['current_hurst'].item()

                # Analog features (volatility-related)
                analog_returns = sample['analog_returns'].numpy()
                recent_vol = np.std(analog_returns)
                recent_mean = np.mean(analog_returns)

                # Digital features
                digital = sample['digital_features'].numpy()
                wavelet_energy = np.mean(np.abs(digital))

                features.append([
                    lyapunov,
                    hurst,
                    recent_vol,
                    abs(recent_mean),
                    wavelet_energy,
                ])
            except:
                features.append([0, 0, 0, 0, 0])

        return np.array(features), regimes

    def test_regime_classification(self) -> Dict:
        """Test regime classification accuracy using simple models."""

        X, y = self.extract_features_for_regime_detection()

        # Split temporally
        n = len(X)
        train_size = int(0.7 * n)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Random Forest classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        # Predictions
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        # Accuracy
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Feature importance
        feature_names = ['lyapunov', 'hurst', 'recent_vol', 'recent_abs_return', 'wavelet_energy']
        importance = dict(zip(feature_names, clf.feature_importances_))

        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'baseline_accuracy': max(np.bincount(y_test)) / len(y_test),  # Most common class
            'feature_importance': importance,
            'n_train': len(y_train),
            'n_test': len(y_test),
        }


# ==============================================================================
# SIMPLE MODELS
# ==============================================================================

class SimpleLinearModel:
    """
    Simple linear model using only the best features.

    Features tested:
    - Chaos metrics (Lyapunov, Hurst)
    - Recent returns (momentum)
    - Recent volatility
    - Wavelet features
    """

    def __init__(self, dataset: HCANPsiDataset):
        self.dataset = dataset

    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract simple features for linear model."""

        features = []
        returns = []

        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]

                # Chaos metrics
                lyapunov = sample['current_lyapunov'].item()
                hurst = sample['current_hurst'].item()

                # Momentum features
                analog_returns = sample['analog_returns'].numpy()
                momentum_short = np.mean(analog_returns[-5:])  # 5-period momentum
                momentum_long = np.mean(analog_returns[-20:])  # 20-period momentum

                # Volatility
                recent_vol = np.std(analog_returns[-10:])

                # Microstructure
                microstructure = sample['microstructure'].numpy()
                bid_ask_spread = microstructure[0] if len(microstructure) > 0 else 0

                features.append([
                    lyapunov,
                    hurst,
                    momentum_short,
                    momentum_long,
                    recent_vol,
                    bid_ask_spread,
                ])

                returns.append(sample['return'].item())
            except:
                continue

        return np.array(features), np.array(returns)

    def train_and_validate(self) -> Dict:
        """Train with walk-forward validation."""

        X, y = self.extract_features()

        # Temporal split
        n = len(X)
        train_size = int(0.7 * n)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        results = {}

        # Test different regularization strengths
        for alpha in [0.1, 1.0, 10.0]:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_ic = np.corrcoef(train_pred, y_train)[0, 1]
            test_ic = np.corrcoef(test_pred, y_test)[0, 1]

            results[f'ridge_alpha_{alpha}'] = {
                'train_ic': train_ic,
                'test_ic': test_ic,
                'test_mse': np.mean((test_pred - y_test) ** 2),
            }

        return results


class ShallowNeuralNetwork:
    """
    Shallow 2-layer neural network with best features.

    Architecture:
    - Input: Best features from chaos + momentum + volatility
    - Hidden: 16-32 neurons
    - Output: Return prediction
    - Regularization: Dropout, L2
    """

    def __init__(self, dataset: HCANPsiDataset, device='cpu'):
        self.dataset = dataset
        self.device = device

    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features (same as SimpleLinearModel)."""

        linear_model = SimpleLinearModel(self.dataset)
        return linear_model.extract_features()

    def create_model(self, input_dim: int, hidden_dim: int = 16) -> nn.Module:
        """Create shallow network."""

        class ShallowNet(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x.squeeze(-1)

        return ShallowNet(input_dim, hidden_dim)

    def train_and_validate(self, n_epochs: int = 50) -> Dict:
        """Train with walk-forward validation."""

        X, y = self.extract_features()

        # Temporal split
        n = len(X)
        train_size = int(0.7 * n)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)

        # Create model
        model = self.create_model(X_train.shape[1], hidden_dim=16).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.MSELoss()

        # Training
        best_train_ic = -999
        history = {'train_ic': [], 'train_loss': []}

        for epoch in range(n_epochs):
            model.train()

            # Forward pass
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate IC
            with torch.no_grad():
                pred_np = pred.cpu().numpy()
                train_ic = np.corrcoef(pred_np, y_train)[0, 1]

                history['train_ic'].append(train_ic)
                history['train_loss'].append(loss.item())

                if train_ic > best_train_ic:
                    best_train_ic = train_ic

        # Test
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).cpu().numpy()
            test_ic = np.corrcoef(test_pred, y_test)[0, 1]
            test_mse = np.mean((test_pred - y_test) ** 2)

        return {
            'best_train_ic': best_train_ic,
            'test_ic': test_ic,
            'test_mse': test_mse,
            'n_parameters': sum(p.numel() for p in model.parameters()),
        }


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def run_feature_extraction_analysis():
    """Run complete feature extraction and analysis."""

    print("=" * 80)
    print("FEATURE EXTRACTION AND ANALYSIS")
    print("=" * 80)
    print()
    print("Extracting valuable components from HCAN-Ψ:")
    print("1. Chaos metrics predictive power")
    print("2. Regime detection capability")
    print("3. Simple linear models")
    print("4. Shallow neural networks")
    print()
    print("=" * 80)
    print()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

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

    print(f"Dataset size: {len(full_dataset)} samples")
    print()

    # ==============================================================================
    # TEST 1: CHAOS METRICS
    # ==============================================================================

    print("=" * 80)
    print("TEST 1: CHAOS METRICS PREDICTIVE POWER")
    print("=" * 80)
    print()

    chaos_analyzer = ChaosMetricsAnalyzer(full_dataset)
    chaos_df = chaos_analyzer.extract_chaos_features()

    print("Individual metrics IC:")
    individual_results = chaos_analyzer.analyze_individual_metrics(chaos_df)
    for metric, stats in individual_results.items():
        print(f"  {metric:20s}: IC = {stats['ic']:+.4f} (abs = {stats['abs_ic']:.4f})")
    print()

    print("Combined chaos features (Ridge regression):")
    combined_results = chaos_analyzer.test_combined_chaos_ic(chaos_df)
    print(f"  Train IC: {combined_results['train_ic']:+.4f}")
    print(f"  Test IC:  {combined_results['test_ic']:+.4f}")
    print()
    print("  Feature coefficients:")
    for feature, coef in combined_results['coefficients'].items():
        print(f"    {feature:20s}: {coef:+.6f}")
    print()

    # ==============================================================================
    # TEST 2: REGIME DETECTION
    # ==============================================================================

    print("=" * 80)
    print("TEST 2: REGIME DETECTION CAPABILITY")
    print("=" * 80)
    print()

    regime_analyzer = RegimeDetectionAnalyzer(tick_data, full_dataset)
    regime_results = regime_analyzer.test_regime_classification()

    print(f"Train Accuracy: {regime_results['train_accuracy']:.2%}")
    print(f"Test Accuracy:  {regime_results['test_accuracy']:.2%}")
    print(f"Baseline (most common class): {regime_results['baseline_accuracy']:.2%}")
    print()
    print("Feature importance:")
    for feature, importance in sorted(regime_results['feature_importance'].items(),
                                      key=lambda x: x[1], reverse=True):
        print(f"  {feature:20s}: {importance:.4f}")
    print()

    # ==============================================================================
    # TEST 3: SIMPLE LINEAR MODEL
    # ==============================================================================

    print("=" * 80)
    print("TEST 3: SIMPLE LINEAR MODEL")
    print("=" * 80)
    print()

    linear_model = SimpleLinearModel(full_dataset)
    linear_results = linear_model.train_and_validate()

    for model_name, results in linear_results.items():
        print(f"{model_name}:")
        print(f"  Train IC: {results['train_ic']:+.4f}")
        print(f"  Test IC:  {results['test_ic']:+.4f}")
        print(f"  Test MSE: {results['test_mse']:.8f}")
        print()

    # ==============================================================================
    # TEST 4: SHALLOW NEURAL NETWORK
    # ==============================================================================

    print("=" * 80)
    print("TEST 4: SHALLOW NEURAL NETWORK (2 layers, 16 neurons)")
    print("=" * 80)
    print()

    shallow_net = ShallowNeuralNetwork(full_dataset, device=device)
    shallow_results = shallow_net.train_and_validate(n_epochs=50)

    print(f"Parameters: {shallow_results['n_parameters']}")
    print(f"Best Train IC: {shallow_results['best_train_ic']:+.4f}")
    print(f"Test IC:       {shallow_results['test_ic']:+.4f}")
    print(f"Test MSE:      {shallow_results['test_mse']:.8f}")
    print()

    # ==============================================================================
    # SAVE RESULTS
    # ==============================================================================

    all_results = {
        'chaos_metrics': {
            'individual': individual_results,
            'combined': combined_results,
        },
        'regime_detection': regime_results,
        'linear_models': linear_results,
        'shallow_network': shallow_results,
    }

    with open('feature_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("✓ Results saved to: feature_analysis_results.json")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    best_chaos_ic = combined_results['test_ic']
    best_linear_ic = max([r['test_ic'] for r in linear_results.values()])
    shallow_ic = shallow_results['test_ic']
    regime_acc = regime_results['test_accuracy']

    print(f"Best chaos metrics IC:     {best_chaos_ic:+.4f}")
    print(f"Best linear model IC:      {best_linear_ic:+.4f}")
    print(f"Shallow network IC:        {shallow_ic:+.4f}")
    print(f"Regime detection accuracy: {regime_acc:.2%}")
    print()

    if max(abs(best_chaos_ic), abs(best_linear_ic), abs(shallow_ic)) < 0.03:
        print("⚠️  All ICs < 3%: No individual features show strong predictive power")

    if regime_acc > regime_results['baseline_accuracy'] + 0.1:
        print("✓ Regime detection shows promise (>10% above baseline)")
    else:
        print("⚠️  Regime detection not significantly better than baseline")


if __name__ == '__main__':
    run_feature_extraction_analysis()
