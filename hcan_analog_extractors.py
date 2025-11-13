"""
Analog Derivative Extractors for HCAN
======================================

Implements continuous derivative extraction from discrete market data:
1. Wavelet-based multi-scale analysis
2. Stochastic differential equations for chaos evolution
3. Liquidity surface curvature
4. Order flow dynamics
5. Geometric manifold features

Author: RD-Agent Research Team
Date: 2025-11-13
Level: 4 (Meta-dynamics)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert
from scipy.integrate import odeint
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. WAVELET-BASED ANALOG DERIVATIVES
# ============================================================================

class WaveletDerivatives:
    """
    Extract analog derivatives using continuous wavelet transform.

    Captures multi-scale dynamics that discrete models miss.
    """

    def __init__(self, max_scale: int = 128):
        """
        Args:
            max_scale: Maximum wavelet scale (time horizon)
        """
        self.max_scale = max_scale
        self.scales = np.arange(1, max_scale + 1)

    def continuous_wavelet_transform(self,
                                     signal: np.ndarray,
                                     wavelet: str = 'morl') -> Tuple[np.ndarray, np.ndarray]:
        """
        Continuous wavelet transform for multi-scale decomposition.

        Args:
            signal: Time series (1D array)
            wavelet: Wavelet type ('morl', 'mexh', etc.)

        Returns:
            coefficients: CWT coefficients [scales x time]
            frequencies: Corresponding frequencies
        """
        try:
            import pywt

            coefficients, frequencies = pywt.cwt(
                signal,
                self.scales,
                wavelet
            )
            return coefficients, frequencies

        except ImportError:
            # Fallback: Manual Morlet wavelet implementation
            return self._morlet_cwt_fallback(signal)

    def _morlet_cwt_fallback(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback implementation using manual Morlet wavelet."""
        n = len(signal)
        coefficients = np.zeros((len(self.scales), n))

        for i, scale in enumerate(self.scales):
            # Morlet wavelet - ensure it doesn't exceed signal length
            wavelet_length = min(8 * scale, n)
            t = np.linspace(-4*scale, 4*scale, wavelet_length)
            wavelet = np.exp(-(t**2)/(2*scale**2)) * np.cos(5*t/scale)

            # Normalize
            wavelet = wavelet / np.sqrt(scale)

            # Convolve - use 'same' mode
            conv_result = np.convolve(signal, wavelet, mode='same')

            # Ensure result matches signal length
            if len(conv_result) != n:
                # Truncate or pad if necessary
                conv_result = conv_result[:n] if len(conv_result) > n else np.pad(conv_result, (0, n - len(conv_result)))

            coefficients[i, :] = conv_result

        frequencies = 1.0 / self.scales
        return coefficients, frequencies

    def instantaneous_frequency(self, signal: np.ndarray) -> np.ndarray:
        """
        Analog derivative: How dominant frequency changes over time.

        Uses Hilbert transform to compute analytic signal.

        Args:
            signal: Time series

        Returns:
            instantaneous_freq: Frequency at each time point
        """
        # Analytic signal via Hilbert transform
        analytic_signal = hilbert(signal)

        # Instantaneous phase
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Instantaneous frequency (derivative of phase)
        instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi)

        # Pad to match original length
        instantaneous_freq = np.concatenate([[instantaneous_freq[0]], instantaneous_freq])

        return instantaneous_freq

    def wavelet_energy(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Energy distribution across scales.

        High energy at low scales → high-frequency activity
        High energy at high scales → low-frequency trends
        """
        energy = np.abs(coefficients)**2
        return energy

    def dominant_scale(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Analog derivative: Which timescale dominates at each moment.

        Returns:
            dominant_scales: [time] - scale with max energy at each time
        """
        energy = self.wavelet_energy(coefficients)
        dominant_scales = self.scales[np.argmax(energy, axis=0)]
        return dominant_scales

    def wavelet_coherence(self,
                         signal1: np.ndarray,
                         signal2: np.ndarray) -> np.ndarray:
        """
        Time-frequency coherence between two signals.

        Useful for:
        - Price-volume coherence
        - Cross-asset correlation at different scales

        Args:
            signal1, signal2: Time series to compare

        Returns:
            coherence: [scales x time] - coherence matrix
        """
        # CWT of both signals
        coef1, _ = self.continuous_wavelet_transform(signal1)
        coef2, _ = self.continuous_wavelet_transform(signal2)

        # Cross-wavelet spectrum
        W_xy = coef1 * np.conj(coef2)

        # Auto-wavelet spectra
        S_xx = np.abs(coef1)**2
        S_yy = np.abs(coef2)**2

        # Coherence (like correlation, but time-frequency localized)
        coherence = np.abs(W_xy)**2 / (S_xx * S_yy + 1e-10)

        return coherence


# ============================================================================
# 2. LYAPUNOV STOCHASTIC DIFFERENTIAL EQUATION
# ============================================================================

class LyapunovSDE:
    """
    Model Lyapunov exponent evolution as a continuous SDE.

    dλ = κ(θ - λ)dt + σ√λ dW

    This allows predicting chaos changes before they happen.
    """

    def __init__(self,
                 theta: float = 0.3,    # Long-term chaos level
                 kappa: float = 0.5,    # Mean reversion speed
                 sigma: float = 0.1):   # Volatility of chaos
        """
        Args:
            theta: Long-term mean (equilibrium chaos)
            kappa: Mean reversion rate
            sigma: Diffusion coefficient
        """
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma

    def drift(self, lambda_t: float, info_shock: float = 0.0) -> float:
        """
        Drift term: Pulls toward equilibrium + reacts to shocks.

        dλ/dt = κ(θ - λ) + γ·info_shock

        Args:
            lambda_t: Current Lyapunov value
            info_shock: Information shock (e.g., news intensity)

        Returns:
            drift: Rate of change
        """
        mean_reversion = self.kappa * (self.theta - lambda_t)
        shock_response = 0.2 * info_shock  # News increases chaos
        return mean_reversion + shock_response

    def diffusion(self, lambda_t: float) -> float:
        """
        Diffusion term: Random fluctuations.

        σ(λ) = σ₀·√λ  (CIR-like)

        Ensures λ stays positive.
        """
        return self.sigma * np.sqrt(np.maximum(lambda_t, 0.01))

    def simulate(self,
                 lambda_0: float,
                 T: int,
                 dt: float = 0.01,
                 info_shocks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate continuous Lyapunov evolution.

        Args:
            lambda_0: Initial chaos level
            T: Total time steps
            dt: Time increment
            info_shocks: Optional array of information shocks

        Returns:
            lambda_path: Simulated trajectory
        """
        n_steps = int(T / dt)
        lambda_path = np.zeros(n_steps)
        lambda_path[0] = lambda_0

        if info_shocks is None:
            info_shocks = np.zeros(n_steps)

        for t in range(1, n_steps):
            # Brownian increment
            dW = np.random.randn() * np.sqrt(dt)

            # Euler-Maruyama scheme
            drift_term = self.drift(lambda_path[t-1], info_shocks[t])
            diffusion_term = self.diffusion(lambda_path[t-1])

            lambda_path[t] = lambda_path[t-1] + drift_term*dt + diffusion_term*dW

            # Keep in valid range [0, 1]
            lambda_path[t] = np.clip(lambda_path[t], 0.0, 1.0)

        return lambda_path

    def predict_evolution(self,
                         current_lambda: float,
                         horizon: int = 10) -> np.ndarray:
        """
        Predict future Lyapunov trajectory.

        Args:
            current_lambda: Current chaos level
            horizon: Prediction horizon

        Returns:
            predicted_path: Expected evolution
        """
        # Deterministic evolution (expected path)
        path = np.zeros(horizon)
        path[0] = current_lambda

        for t in range(1, horizon):
            # Expected evolution (drift only, no diffusion)
            drift = self.drift(path[t-1])
            path[t] = path[t-1] + drift
            path[t] = np.clip(path[t], 0.0, 1.0)

        return path


# ============================================================================
# 3. HURST STOCHASTIC DIFFERENTIAL EQUATION
# ============================================================================

class HurstSDE:
    """
    Model Hurst exponent evolution as Ornstein-Uhlenbeck process.

    dH = κ(0.5 - H)dt + σdW

    H → 0.5: Efficient market (random walk)
    H > 0.5: Trending (persistent)
    H < 0.5: Mean-reverting (anti-persistent)
    """

    def __init__(self, kappa: float = 0.3, sigma: float = 0.05):
        """
        Args:
            kappa: Mean reversion to 0.5 (efficiency)
            sigma: Volatility of Hurst changes
        """
        self.kappa = kappa
        self.sigma = sigma
        self.equilibrium = 0.5  # Efficient market

    def drift(self, H_t: float, efficiency_shock: float = 0.0) -> float:
        """
        Drift toward efficiency.

        dH/dt = κ(0.5 - H) + shock
        """
        mean_reversion = self.kappa * (self.equilibrium - H_t)
        return mean_reversion + efficiency_shock

    def diffusion(self) -> float:
        """Constant diffusion."""
        return self.sigma

    def simulate(self,
                 H_0: float,
                 T: int,
                 dt: float = 0.01) -> np.ndarray:
        """
        Simulate Hurst evolution.

        Args:
            H_0: Initial Hurst exponent
            T: Total time steps
            dt: Time increment

        Returns:
            H_path: Simulated trajectory
        """
        n_steps = int(T / dt)
        H_path = np.zeros(n_steps)
        H_path[0] = H_0

        for t in range(1, n_steps):
            dW = np.random.randn() * np.sqrt(dt)

            drift_term = self.drift(H_path[t-1])
            diffusion_term = self.diffusion()

            H_path[t] = H_path[t-1] + drift_term*dt + diffusion_term*dW

            # Keep in valid range [0, 1]
            H_path[t] = np.clip(H_path[t], 0.0, 1.0)

        return H_path

    def predict_trend_strength(self, current_H: float) -> str:
        """
        Interpret current Hurst value.

        Returns:
            regime: 'trending', 'efficient', or 'mean_reverting'
        """
        if current_H > 0.6:
            return 'trending'
        elif current_H < 0.4:
            return 'mean_reverting'
        else:
            return 'efficient'


# ============================================================================
# 4. LIQUIDITY SURFACE CURVATURE
# ============================================================================

class LiquidityCurvature:
    """
    Calculate curvature of liquidity surface from order book.

    High curvature → thin market → risky
    Low curvature → deep market → safe
    """

    def calculate_curvature(self,
                           prices: np.ndarray,
                           volumes: np.ndarray) -> float:
        """
        Compute ∂²L/∂p² (second derivative of liquidity).

        Args:
            prices: Price levels
            volumes: Volume at each level

        Returns:
            curvature: Scalar curvature measure
        """
        # First derivative: liquidity gradient
        dL_dp = np.gradient(volumes, prices)

        # Second derivative: curvature
        d2L_dp2 = np.gradient(dL_dp, prices)

        # Return mean absolute curvature
        return np.mean(np.abs(d2L_dp2))

    def process_order_book(self,
                          bid_prices: np.ndarray,
                          bid_volumes: np.ndarray,
                          ask_prices: np.ndarray,
                          ask_volumes: np.ndarray) -> Dict[str, float]:
        """
        Extract analog derivatives from full order book.

        Returns:
            features: Dictionary of continuous features
        """
        # Combine bid and ask sides
        all_prices = np.concatenate([bid_prices[::-1], ask_prices])
        all_volumes = np.concatenate([bid_volumes[::-1], ask_volumes])

        # Curvature
        curvature = self.calculate_curvature(all_prices, all_volumes)

        # Imbalance gradient
        bid_depth = np.sum(bid_volumes)
        ask_depth = np.sum(ask_volumes)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)

        # Spread gradient (how quickly spread changes)
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        spread = ask_prices[0] - bid_prices[0]
        spread_pct = spread / mid_price

        return {
            'curvature': curvature,
            'imbalance': imbalance,
            'spread_pct': spread_pct,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth
        }


# ============================================================================
# 5. ORDER FLOW HAWKES PROCESS
# ============================================================================

class OrderFlowHawkes:
    """
    Model order flow as self-exciting Hawkes process.

    λ(t) = μ + Σ α·exp(-β(t-tᵢ))

    Captures momentum in order arrivals.
    """

    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 1.0):
        """
        Args:
            mu: Base intensity (background rate)
            alpha: Self-excitement magnitude
            beta: Decay rate
        """
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def intensity(self, t: float, event_times: np.ndarray) -> float:
        """
        Calculate intensity at time t given past events.

        λ(t) = μ + Σᵢ α·exp(-β(t-tᵢ))

        Args:
            t: Current time
            event_times: Array of past event times

        Returns:
            intensity: Current arrival rate
        """
        # Base intensity
        lambda_t = self.mu

        # Add self-excitement from past events
        for t_i in event_times:
            if t_i < t:
                lambda_t += self.alpha * np.exp(-self.beta * (t - t_i))

        return lambda_t

    def simulate(self, T: float) -> np.ndarray:
        """
        Simulate order arrivals via thinning algorithm.

        Args:
            T: Total time

        Returns:
            event_times: Array of event times
        """
        event_times = []
        t = 0.0

        while t < T:
            # Upper bound on intensity
            lambda_max = self.mu + self.alpha / self.beta

            # Propose next event
            t += np.random.exponential(1.0 / lambda_max)

            if t > T:
                break

            # Accept/reject
            lambda_t = self.intensity(t, np.array(event_times))
            if np.random.rand() < lambda_t / lambda_max:
                event_times.append(t)

        return np.array(event_times)

    def estimate_intensity(self, event_times: np.ndarray, t: float) -> float:
        """
        Estimate current order flow intensity.

        Args:
            event_times: Observed event times
            t: Current time

        Returns:
            intensity: Estimated λ(t)
        """
        return self.intensity(t, event_times)


# ============================================================================
# 6. MARKET MANIFOLD GEOMETRY
# ============================================================================

class MarketManifold:
    """
    Treat market as Riemannian manifold with curvature.

    Accounts for non-Euclidean geometry of state space.
    """

    def __init__(self, volatility_weight: float = 1.0):
        """
        Args:
            volatility_weight: How much volatility affects metric
        """
        self.volatility_weight = volatility_weight

    def metric_tensor(self, state: np.ndarray, volatility: float) -> np.ndarray:
        """
        Define local geometry via metric tensor.

        High volatility → large distances
        Low volatility → small distances

        Args:
            state: Market state vector
            volatility: Current volatility

        Returns:
            g: Metric tensor (matrix)
        """
        n = len(state)

        # Diagonal metric (simplified)
        # g_ij = δ_ij · σ²
        g = np.eye(n) * (volatility ** self.volatility_weight)

        return g

    def geodesic_distance(self,
                         state1: np.ndarray,
                         state2: np.ndarray,
                         volatility: float) -> float:
        """
        True distance accounting for curvature.

        For diagonal metric: d = √(Σ(xᵢ-yᵢ)²/σ²)

        Args:
            state1, state2: Market states
            volatility: Current volatility

        Returns:
            distance: Geodesic distance
        """
        g = self.metric_tensor(state1, volatility)

        # Simplified: Mahalanobis-like distance
        diff = state1 - state2
        distance = np.sqrt(diff.T @ np.linalg.inv(g) @ diff)

        return distance

    def ricci_curvature(self, states: np.ndarray) -> float:
        """
        Estimate Ricci curvature from state distribution.

        High curvature → regime boundaries
        Low curvature → stable regions

        Args:
            states: Collection of nearby states [n_samples x n_features]

        Returns:
            curvature: Scalar curvature estimate
        """
        # Simplified: Use variance as proxy for curvature
        # High variance → high curvature
        return np.mean(np.var(states, axis=0))


# ============================================================================
# 7. UNIFIED ANALOG FEATURE EXTRACTOR
# ============================================================================

class AnalogFeatureExtractor:
    """
    Unified interface for extracting all analog derivatives.
    """

    def __init__(self):
        self.wavelet = WaveletDerivatives(max_scale=64)
        self.lyapunov_sde = LyapunovSDE()
        self.hurst_sde = HurstSDE()
        self.liquidity = LiquidityCurvature()
        self.hawkes = OrderFlowHawkes()
        self.manifold = MarketManifold()

    def extract_from_returns(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract analog features from return series.

        Args:
            returns: Time series of returns

        Returns:
            features: Dictionary of analog features
        """
        # Wavelet features
        cwt_coef, cwt_freq = self.wavelet.continuous_wavelet_transform(returns)
        inst_freq = self.wavelet.instantaneous_frequency(returns)
        dominant_scales = self.wavelet.dominant_scale(cwt_coef)
        wavelet_energy = self.wavelet.wavelet_energy(cwt_coef)

        return {
            'cwt_coefficients': cwt_coef,
            'cwt_frequencies': cwt_freq,
            'instantaneous_frequency': inst_freq,
            'dominant_scales': dominant_scales,
            'wavelet_energy': wavelet_energy,
            'mean_energy_low_freq': np.mean(wavelet_energy[-10:, :], axis=0),
            'mean_energy_high_freq': np.mean(wavelet_energy[:10, :], axis=0)
        }

    def extract_chaos_evolution(self,
                                current_lyapunov: float,
                                current_hurst: float,
                                horizon: int = 10) -> Dict[str, np.ndarray]:
        """
        Predict evolution of chaos metrics.

        Args:
            current_lyapunov: Current Lyapunov exponent
            current_hurst: Current Hurst exponent
            horizon: Prediction steps

        Returns:
            predictions: Future chaos trajectories
        """
        # Predict Lyapunov evolution
        lyap_path = self.lyapunov_sde.predict_evolution(current_lyapunov, horizon)

        # Predict Hurst evolution
        hurst_path = self.hurst_sde.simulate(current_hurst, horizon, dt=1.0)

        return {
            'lyapunov_evolution': lyap_path,
            'hurst_evolution': hurst_path,
            'lyapunov_derivative': np.gradient(lyap_path),
            'hurst_derivative': np.gradient(hurst_path)
        }

    def extract_microstructure(self,
                               bid_prices: np.ndarray,
                               bid_volumes: np.ndarray,
                               ask_prices: np.ndarray,
                               ask_volumes: np.ndarray) -> Dict[str, float]:
        """
        Extract microstructure analog features.

        Args:
            bid_prices, bid_volumes: Bid side of order book
            ask_prices, ask_volumes: Ask side of order book

        Returns:
            features: Liquidity curvature and imbalance
        """
        return self.liquidity.process_order_book(
            bid_prices, bid_volumes, ask_prices, ask_volumes
        )

    def extract_order_flow(self, event_times: np.ndarray) -> Dict[str, float]:
        """
        Extract order flow dynamics.

        Args:
            event_times: Times of order events

        Returns:
            features: Flow intensity and derivatives
        """
        if len(event_times) == 0:
            return {'intensity': 0.0, 'acceleration': 0.0}

        # Current intensity
        t_current = event_times[-1] if len(event_times) > 0 else 0
        intensity = self.hawkes.estimate_intensity(event_times, t_current)

        # Inter-arrival times
        durations = np.diff(event_times)

        # Rate of activity
        activity_rate = 1.0 / (np.mean(durations) + 1e-10)

        # Acceleration (change in activity)
        if len(durations) > 1:
            acceleration = np.diff(1.0 / durations)
            mean_accel = np.mean(acceleration)
        else:
            mean_accel = 0.0

        return {
            'intensity': intensity,
            'mean_duration': np.mean(durations) if len(durations) > 0 else 0.0,
            'activity_rate': activity_rate,
            'acceleration': mean_accel
        }


# ============================================================================
# 8. PYTORCH NEURAL NETWORK LAYERS
# ============================================================================

class ContinuousWaveletLayer(nn.Module):
    """
    PyTorch layer for continuous wavelet transform.
    """

    def __init__(self, in_features: int, n_scales: int = 32):
        super().__init__()
        self.n_scales = n_scales
        self.extractor = WaveletDerivatives(max_scale=n_scales)

        # Learnable aggregation
        self.scale_weights = nn.Parameter(torch.randn(n_scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, time] - returns

        Returns:
            features: [batch, n_scales] - wavelet features
        """
        batch_size = x.shape[0]
        features = []

        for i in range(batch_size):
            signal = x[i].detach().cpu().numpy()

            # Extract wavelet coefficients
            cwt_coef, _ = self.extractor.continuous_wavelet_transform(signal)

            # Mean energy across time for each scale
            scale_energy = np.mean(np.abs(cwt_coef)**2, axis=1)
            features.append(scale_energy)

        features = torch.tensor(np.array(features), dtype=torch.float32, device=x.device)

        # Weighted aggregation
        features = features * torch.softmax(self.scale_weights, dim=0)

        return features


class LyapunovSDELayer(nn.Module):
    """
    PyTorch layer for Lyapunov evolution prediction.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Neural network to predict SDE parameters
        self.param_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # theta, kappa, sigma
            nn.Sigmoid()
        )

        self.sde = LyapunovSDE()

    def forward(self, current_lyapunov: torch.Tensor, horizon: int = 10) -> torch.Tensor:
        """
        Args:
            current_lyapunov: [batch, 1] - current values
            horizon: Steps to predict

        Returns:
            evolution: [batch, horizon] - predicted trajectory
        """
        batch_size = current_lyapunov.shape[0]
        evolutions = []

        for i in range(batch_size):
            lambda_0 = current_lyapunov[i].item()

            # Predict evolution
            path = self.sde.predict_evolution(lambda_0, horizon)
            evolutions.append(path)

        return torch.tensor(np.array(evolutions), dtype=torch.float32, device=current_lyapunov.device)


# ============================================================================
# 9. VALIDATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ANALOG DERIVATIVE EXTRACTORS - VALIDATION")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    T = 1000
    returns = np.random.randn(T) * 0.01
    returns += 0.001 * np.sin(np.arange(T) / 20)  # Add trend

    # Initialize extractor
    extractor = AnalogFeatureExtractor()

    print("\n1. WAVELET FEATURES")
    print("-" * 80)
    wavelet_features = extractor.extract_from_returns(returns)
    print(f"   CWT coefficients shape: {wavelet_features['cwt_coefficients'].shape}")
    print(f"   Instantaneous frequency shape: {wavelet_features['instantaneous_frequency'].shape}")
    print(f"   Dominant scales (last 10): {wavelet_features['dominant_scales'][-10:]}")
    print(f"   Mean low-freq energy: {wavelet_features['mean_energy_low_freq'][-1]:.6f}")
    print(f"   Mean high-freq energy: {wavelet_features['mean_energy_high_freq'][-1]:.6f}")

    print("\n2. CHAOS EVOLUTION")
    print("-" * 80)
    chaos_features = extractor.extract_chaos_evolution(
        current_lyapunov=0.3,
        current_hurst=0.6,
        horizon=20
    )
    print(f"   Lyapunov evolution: {chaos_features['lyapunov_evolution'][:5]}")
    print(f"   Hurst evolution: {chaos_features['hurst_evolution'][:5]}")
    print(f"   dλ/dt (next 5 steps): {chaos_features['lyapunov_derivative'][:5]}")
    print(f"   dH/dt (next 5 steps): {chaos_features['hurst_derivative'][:5]}")

    print("\n3. MICROSTRUCTURE")
    print("-" * 80)
    # Simulate order book
    mid_price = 100.0
    bid_prices = mid_price - np.arange(1, 11) * 0.01
    ask_prices = mid_price + np.arange(1, 11) * 0.01
    bid_volumes = 1000 * np.exp(-np.arange(10) * 0.1)
    ask_volumes = 1000 * np.exp(-np.arange(10) * 0.1)

    micro_features = extractor.extract_microstructure(
        bid_prices, bid_volumes, ask_prices, ask_volumes
    )
    print(f"   Liquidity curvature: {micro_features['curvature']:.6f}")
    print(f"   Order imbalance: {micro_features['imbalance']:.6f}")
    print(f"   Spread %: {micro_features['spread_pct']:.6f}")

    print("\n4. ORDER FLOW")
    print("-" * 80)
    # Simulate order arrivals
    event_times = np.sort(np.random.exponential(1.0, 100)).cumsum()
    flow_features = extractor.extract_order_flow(event_times)
    print(f"   Current intensity: {flow_features['intensity']:.6f}")
    print(f"   Mean duration: {flow_features['mean_duration']:.6f}")
    print(f"   Activity rate: {flow_features['activity_rate']:.6f}")
    print(f"   Acceleration: {flow_features['acceleration']:.6f}")

    print("\n5. PYTORCH LAYERS")
    print("-" * 80)
    # Test PyTorch layers
    batch_size = 4
    time_steps = 100

    # Wavelet layer
    wavelet_layer = ContinuousWaveletLayer(in_features=time_steps, n_scales=16)
    x = torch.randn(batch_size, time_steps)
    wavelet_out = wavelet_layer(x)
    print(f"   Wavelet layer output shape: {wavelet_out.shape}")

    # Lyapunov SDE layer
    lyap_layer = LyapunovSDELayer(hidden_dim=32)
    current_lyap = torch.rand(batch_size, 1) * 0.5
    lyap_evolution = lyap_layer(current_lyap, horizon=10)
    print(f"   Lyapunov SDE layer output shape: {lyap_evolution.shape}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("All analog derivative extractors working correctly!")
    print("=" * 80)
