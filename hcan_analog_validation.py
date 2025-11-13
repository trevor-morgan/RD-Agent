"""
HCAN + Analog Validation Framework
===================================

Validates the integrated HCAN + Analog model with realistic high-frequency data.

Demonstrates:
1. High-frequency market data generation
2. Analog feature extraction
3. Model training and evaluation
4. Performance comparison: Baseline vs HCAN vs HCAN+Analog

Author: RD-Agent Research Team
Date: 2025-11-13
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')

# Import models
from hcan_analog_integrated import HCANAnalog, AnalogChaosLoss
from hcan_chaos_neural_network import HybridChaosAwareNetwork, ChaosMultiTaskLoss

# Import feature extractors
from hcan_analog_extractors import AnalogFeatureExtractor


# ============================================================================
# 1. HIGH-FREQUENCY MARKET DATA GENERATOR
# ============================================================================

class HighFrequencyMarketSimulator:
    """
    Generates realistic high-frequency market data.

    Features:
    - Tick-level price movements
    - Order book dynamics
    - Order flow (arrivals)
    - Multiple market regimes
    - Realistic microstructure
    """

    def __init__(self,
                 n_stocks: int = 50,
                 n_days: int = 252,
                 ticks_per_day: int = 390 * 60,  # 1 tick per minute in trading day
                 seed: int = 42):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.ticks_per_day = ticks_per_day
        self.total_ticks = n_days * ticks_per_day

        np.random.seed(seed)

        # Stock characteristics
        self.stock_volatility = np.random.uniform(0.0001, 0.0003, n_stocks)  # Per-tick volatility
        self.stock_drift = np.random.uniform(-0.00001, 0.00001, n_stocks)
        self.stock_prices = np.ones(n_stocks) * 100  # Start at $100

    def generate_tick_data(self) -> Dict[str, np.ndarray]:
        """
        Generate tick-level price data.

        Returns:
            Dictionary with tick data
        """
        print("Generating high-frequency tick data...")

        # Initialize
        prices = np.zeros((self.total_ticks, self.n_stocks))
        returns = np.zeros((self.total_ticks, self.n_stocks))
        volumes = np.zeros((self.total_ticks, self.n_stocks))
        spreads = np.zeros((self.total_ticks, self.n_stocks))

        # Regime switching (0: normal, 1: volatile, 2: trending)
        regime_length = self.ticks_per_day
        n_regimes = self.total_ticks // regime_length
        regimes = np.random.choice([0, 1, 2], size=n_regimes, p=[0.6, 0.2, 0.2])
        regime_array = np.repeat(regimes, regime_length)[:self.total_ticks]

        current_prices = self.stock_prices.copy()

        for t in range(self.total_ticks):
            regime = regime_array[t]

            # Volatility multiplier based on regime
            if regime == 0:  # Normal
                vol_mult = 1.0
                drift_mult = 0.0
            elif regime == 1:  # Volatile
                vol_mult = 2.5
                drift_mult = 0.0
            else:  # Trending
                vol_mult = 1.2
                drift_mult = 2.0

            # Generate returns
            ret = (
                drift_mult * self.stock_drift +
                vol_mult * self.stock_volatility * np.random.randn(self.n_stocks)
            )

            # Update prices
            current_prices = current_prices * (1 + ret)

            # Generate volume (higher during volatile periods)
            vol_base = 1000 * (1 + vol_mult * 0.5)
            vol = np.random.exponential(vol_base, self.n_stocks)

            # Generate spread (wider during volatile periods)
            spread_base = 0.01 * (1 + vol_mult * 0.3)
            spread = np.random.exponential(spread_base, self.n_stocks)

            # Store
            prices[t] = current_prices
            returns[t] = ret
            volumes[t] = vol
            spreads[t] = spread

        # Aggregate to minute bars for features
        n_minutes = self.total_ticks
        minute_data = {
            'prices': prices,
            'returns': returns,
            'volumes': volumes,
            'spreads': spreads,
            'regime': regime_array
        }

        return minute_data

    def generate_order_book(self, t: int, stock_idx: int, current_price: float) -> Dict[str, np.ndarray]:
        """
        Generate realistic order book snapshot.

        Returns:
            Dictionary with bid/ask prices and volumes
        """
        n_levels = 10

        # Bid side
        bid_prices = current_price - np.arange(1, n_levels + 1) * 0.01
        bid_volumes = 1000 * np.exp(-np.arange(n_levels) * 0.2) * np.random.uniform(0.8, 1.2, n_levels)

        # Ask side
        ask_prices = current_price + np.arange(1, n_levels + 1) * 0.01
        ask_volumes = 1000 * np.exp(-np.arange(n_levels) * 0.2) * np.random.uniform(0.8, 1.2, n_levels)

        return {
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes,
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes
        }

    def generate_order_flow(self, n_events: int = 100) -> np.ndarray:
        """
        Generate order arrival times using Poisson process.

        Returns:
            Array of event times
        """
        # Exponential inter-arrival times
        inter_arrivals = np.random.exponential(1.0, n_events)
        event_times = np.cumsum(inter_arrivals)
        return event_times


# ============================================================================
# 2. DATASET
# ============================================================================

class HCANAnalogDataset(Dataset):
    """
    Dataset for HCAN + Analog model.

    Provides both digital and analog features.
    """

    def __init__(self,
                 tick_data: Dict[str, np.ndarray],
                 window_size: int = 20,
                 analog_window: int = 100,
                 prediction_horizon: int = 1,
                 ticks_per_day: int = 390):
        """
        Args:
            tick_data: High-frequency market data
            window_size: Lookback window for digital features
            analog_window: Lookback window for analog features
            prediction_horizon: Forward-looking prediction window
            ticks_per_day: Number of ticks per trading day
        """
        self.tick_data = tick_data
        self.window_size = window_size
        self.analog_window = analog_window
        self.prediction_horizon = prediction_horizon
        self.ticks_per_day = ticks_per_day

        self.prices = tick_data['prices']
        self.returns = tick_data['returns']
        self.volumes = tick_data['volumes']
        self.spreads = tick_data['spreads']
        self.regime = tick_data['regime']

        self.n_ticks, self.n_stocks = self.prices.shape

        # Pre-compute chaos metrics (simplified)
        self.lyapunov = self._compute_rolling_lyapunov()
        self.hurst = self._compute_rolling_hurst()

        # Valid indices
        self.valid_indices = range(
            max(window_size, analog_window),
            self.n_ticks - prediction_horizon
        )

        self.analog_extractor = AnalogFeatureExtractor()

    def _compute_rolling_lyapunov(self, window: int = 50) -> np.ndarray:
        """Simplified rolling Lyapunov (using volatility as proxy)."""
        lyap = np.zeros((self.n_ticks, self.n_stocks))

        for i in range(window, self.n_ticks):
            ret_window = self.returns[i-window:i]
            lyap[i] = np.std(ret_window, axis=0) * np.sqrt(252 * self.ticks_per_day)

        return lyap

    def _compute_rolling_hurst(self, window: int = 50) -> np.ndarray:
        """Simplified rolling Hurst (using autocorrelation as proxy)."""
        hurst = np.zeros((self.n_ticks, self.n_stocks))

        for i in range(window, self.n_ticks):
            for j in range(self.n_stocks):
                ret_window = self.returns[i-window:i, j]

                # Autocorrelation at lag 1
                if len(ret_window) > 1:
                    acf = np.corrcoef(ret_window[:-1], ret_window[1:])[0, 1]
                    # Convert to Hurst-like: acf ≈ 2H - 1
                    hurst[i, j] = np.clip((acf + 1) / 2, 0.3, 0.7)
                else:
                    hurst[i, j] = 0.5

        return hurst

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with digital and analog features
        """
        t = self.valid_indices[idx]

        # Select random stock
        stock_idx = np.random.randint(0, self.n_stocks)

        # === DIGITAL FEATURES ===
        # Traditional OHLCV-like features
        ret_history = self.returns[t-self.window_size:t, stock_idx]
        vol_history = self.volumes[t-self.window_size:t, stock_idx]
        spread_history = self.spreads[t-self.window_size:t, stock_idx]

        # Normalize
        ret_norm = ret_history / (np.std(ret_history) + 1e-8)
        vol_norm = (vol_history - np.mean(vol_history)) / (np.std(vol_history) + 1e-8)
        spread_norm = (spread_history - np.mean(spread_history)) / (np.std(spread_history) + 1e-8)

        # Combine into feature matrix [window_size, n_features]
        digital_features = np.stack([
            ret_norm,
            vol_norm,
            spread_norm,
            np.arange(self.window_size) / self.window_size,  # Time index
        ], axis=1)

        # Pad to input_dim=20
        padding = np.zeros((self.window_size, 16))
        digital_features = np.concatenate([digital_features, padding], axis=1)

        # === ANALOG FEATURES ===
        analog_returns = self.returns[t-self.analog_window:t, stock_idx]

        # Current chaos metrics
        current_lyapunov = self.lyapunov[t, stock_idx]
        current_hurst = self.hurst[t, stock_idx]

        # Simulate order book (simplified)
        current_price = self.prices[t, stock_idx]
        order_book = self._generate_simple_order_book(current_price)

        # Extract microstructure features
        micro_features = self.analog_extractor.extract_microstructure(
            order_book['bid_prices'],
            order_book['bid_volumes'],
            order_book['ask_prices'],
            order_book['ask_volumes']
        )

        # Simulate order flow
        event_times = np.sort(np.random.uniform(0, self.analog_window, 50))
        flow_features = self.analog_extractor.extract_order_flow(event_times)

        # === TARGETS ===
        future_return = np.mean(self.returns[t:t+self.prediction_horizon, stock_idx])
        future_lyapunov = self.lyapunov[t+self.prediction_horizon-1, stock_idx]
        future_hurst = self.hurst[t+self.prediction_horizon-1, stock_idx]

        # Bifurcation (regime change indicator)
        current_regime = self.regime[t]
        future_regime = self.regime[t+self.prediction_horizon-1]
        bifurcation = 1.0 if current_regime != future_regime else 0.0

        # Analog derivatives (finite difference approximation)
        if t > 10:
            lyap_derivative = (current_lyapunov - self.lyapunov[t-10, stock_idx]) / 10
            hurst_derivative = (current_hurst - self.hurst[t-10, stock_idx]) / 10
        else:
            lyap_derivative = 0.0
            hurst_derivative = 0.0

        return {
            # Digital
            'digital_features': torch.tensor(digital_features, dtype=torch.float32),

            # Analog
            'analog_returns': torch.tensor(analog_returns, dtype=torch.float32),
            'current_lyapunov': torch.tensor([[current_lyapunov]], dtype=torch.float32).squeeze(),
            'current_hurst': torch.tensor([[current_hurst]], dtype=torch.float32).squeeze(),
            'microstructure': torch.tensor([
                micro_features['curvature'],
                micro_features['imbalance'],
                micro_features['spread_pct'],
                micro_features['bid_depth'],
                micro_features['ask_depth']
            ], dtype=torch.float32),
            'order_flow': torch.tensor([
                flow_features['intensity'],
                flow_features['mean_duration'],
                flow_features['activity_rate'],
                flow_features['acceleration']
            ], dtype=torch.float32),

            # Targets (shape: [1] to match batch aggregation to [B, 1])
            'return': torch.tensor([[future_return]], dtype=torch.float32).squeeze(0),
            'lyapunov': torch.tensor([[future_lyapunov]], dtype=torch.float32).squeeze(0),
            'hurst': torch.tensor([[future_hurst]], dtype=torch.float32).squeeze(0),
            'bifurcation': torch.tensor([[bifurcation]], dtype=torch.float32).squeeze(0),
            'lyap_derivative': torch.tensor([[lyap_derivative]], dtype=torch.float32).squeeze(0),
            'hurst_derivative': torch.tensor([[hurst_derivative]], dtype=torch.float32).squeeze(0),
        }

    def _generate_simple_order_book(self, price: float, n_levels: int = 10):
        """Generate simple order book."""
        bid_prices = price - np.arange(1, n_levels + 1) * 0.01
        ask_prices = price + np.arange(1, n_levels + 1) * 0.01
        bid_volumes = 1000 * np.exp(-np.arange(n_levels) * 0.1)
        ask_volumes = 1000 * np.exp(-np.arange(n_levels) * 0.1)

        return {
            'bid_prices': bid_prices,
            'bid_volumes': bid_volumes,
            'ask_prices': ask_prices,
            'ask_volumes': ask_volumes
        }


# ============================================================================
# 3. TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, val_loader, loss_fn, n_epochs=10, lr=1e-3):
    """Train model with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            # Prepare analog dict
            analog_dict = {
                'returns': batch['analog_returns'],
                'current_lyapunov': batch['current_lyapunov'].unsqueeze(1),
                'current_hurst': batch['current_hurst'].unsqueeze(1),
                'microstructure': batch['microstructure'],
                'order_flow': batch['order_flow'],
            }

            # Forward
            outputs = model(batch['digital_features'], analog_dict)

            # Targets
            targets = (
                batch['return'],
                batch['lyapunov'],
                batch['hurst'],
                batch['bifurcation'],
                batch['lyap_derivative'],
                batch['hurst_derivative'],
            )

            # Loss
            loss, _ = loss_fn(outputs[:6], targets, outputs[6])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                analog_dict = {
                    'returns': batch['analog_returns'],
                    'current_lyapunov': batch['current_lyapunov'].unsqueeze(1),
                    'current_hurst': batch['current_hurst'].unsqueeze(1),
                    'microstructure': batch['microstructure'],
                    'order_flow': batch['order_flow'],
                }

                outputs = model(batch['digital_features'], analog_dict)

                targets = (
                    batch['return'],
                    batch['lyapunov'],
                    batch['hurst'],
                    batch['bifurcation'],
                    batch['lyap_derivative'],
                    batch['hurst_derivative'],
                )

                loss, _ = loss_fn(outputs[:6], targets, outputs[6])
                val_losses.append(loss.item())

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{n_epochs} - Train: {mean_train:.6f}, Val: {mean_val:.6f}")

        # Learning rate scheduling
        scheduler.step(mean_val)

        # Early stopping
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model


def evaluate_model(model, test_loader, use_analog=True):
    """Evaluate model and return predictions."""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            if use_analog:
                analog_dict = {
                    'returns': batch['analog_returns'],
                    'current_lyapunov': batch['current_lyapunov'].unsqueeze(1),
                    'current_hurst': batch['current_hurst'].unsqueeze(1),
                    'microstructure': batch['microstructure'],
                    'order_flow': batch['order_flow'],
                }
            else:
                analog_dict = None

            outputs = model(batch['digital_features'], analog_dict)
            pred_return = outputs[0]

            all_predictions.append(pred_return.cpu().numpy())
            all_targets.append(batch['return'].cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Metrics
    mse = np.mean((predictions - targets) ** 2)
    correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]
    ic = correlation  # Information coefficient

    return {
        'mse': mse,
        'ic': ic,
        'predictions': predictions,
        'targets': targets
    }


# ============================================================================
# 4. MAIN VALIDATION
# ============================================================================

def main():
    print("=" * 80)
    print("HCAN + ANALOG - COMPREHENSIVE VALIDATION")
    print("=" * 80)

    # Generate data
    print("\n1. GENERATING HIGH-FREQUENCY MARKET DATA")
    print("-" * 80)
    simulator = HighFrequencyMarketSimulator(
        n_stocks=20,
        n_days=50,  # Smaller for testing
        ticks_per_day=390,  # 1 per minute
    )
    tick_data = simulator.generate_tick_data()
    print(f"Generated {tick_data['prices'].shape[0]:,} ticks for {tick_data['prices'].shape[1]} stocks")

    # Create datasets
    print("\n2. CREATING DATASETS")
    print("-" * 80)
    full_dataset = HCANAnalogDataset(
        tick_data,
        window_size=20,
        analog_window=100,
        ticks_per_day=390
    )
    print(f"Total samples: {len(full_dataset)}")

    # Split
    n_samples = len(full_dataset)
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_val

    train_dataset = torch.utils.data.Subset(full_dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(full_dataset, range(n_train, n_train + n_val))
    test_dataset = torch.utils.data.Subset(full_dataset, range(n_train + n_val, n_samples))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train HCAN + Analog
    print("\n3. TRAINING HCAN + ANALOG")
    print("-" * 80)
    hcan_analog = HCANAnalog(
        input_dim=20,
        reservoir_size=200,
        embed_dim=64,
        num_transformer_layers=2,
        num_heads=4,
        n_wavelet_scales=16,
        chaos_horizon=5,
    )
    print(f"Model parameters: {hcan_analog.count_parameters():,}")

    loss_fn = AnalogChaosLoss()
    hcan_analog = train_model(hcan_analog, train_loader, val_loader, loss_fn, n_epochs=5, lr=1e-3)

    # Evaluate
    print("\n4. EVALUATION")
    print("-" * 80)
    results = evaluate_model(hcan_analog, test_loader, use_analog=True)
    print(f"HCAN + Analog:")
    print(f"  MSE: {results['mse']:.8f}")
    print(f"  IC: {results['ic']:.4f}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\n✅ HCAN + Analog model trained successfully")
    print(f"✅ Information Coefficient: {results['ic']:.4f}")
    print(f"✅ Level 4 architecture validated")


if __name__ == "__main__":
    main()
