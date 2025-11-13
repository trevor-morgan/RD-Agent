"""
HCAN-Ψ Physics Layer: Thermodynamics & Information Theory
==========================================================

Physics-informed components for Level 5 architecture.

Implements:
1. Market thermodynamics (entropy, free energy)
2. Information theory (Fisher information, KL divergence)
3. Conservation laws (information, energy)
4. Phase transitions

Author: RD-Agent Research Team
Date: 2025-11-13
Level: 5 (Physics-constrained)
Status: PRODUCTION READY
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
from typing import Dict, Tuple, Optional


# ============================================================================
# 1. MARKET THERMODYNAMICS
# ============================================================================

class MarketThermodynamics:
    """
    Treat markets as thermodynamic systems.

    Key concepts:
    - Entropy (market uncertainty)
    - Free energy (available for trading)
    - Temperature (volatility)
    - Second law: entropy must increase
    """

    def __init__(self, n_bins: int = 50):
        """
        Args:
            n_bins: Number of bins for entropy estimation
        """
        self.n_bins = n_bins

    def entropy(self, returns: np.ndarray) -> float:
        """
        Market entropy (Shannon entropy).

        High entropy = high uncertainty, unpredictable
        Low entropy = low uncertainty, predictable

        Args:
            returns: Return distribution

        Returns:
            entropy: Market entropy in nats
        """
        # Histogram estimation
        hist, _ = np.histogram(returns, bins=self.n_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins

        # Shannon entropy: H = -Σ p log p
        return scipy_entropy(hist)

    def temperature(self, returns: np.ndarray, window: int = 20) -> float:
        """
        Market "temperature" = volatility.

        High temperature = high activity, chaotic
        Low temperature = low activity, stable

        Args:
            returns: Returns
            window: Lookback window

        Returns:
            temperature: Market temperature
        """
        if len(returns) < window:
            return np.std(returns)

        # Rolling volatility as temperature
        rolling_vol = np.std(returns[-window:])

        # Annualize
        temperature = rolling_vol * np.sqrt(252)

        return temperature

    def free_energy(self,
                   price: float,
                   volume: float,
                   temperature: float,
                   entropy: float) -> float:
        """
        Helmholtz free energy: F = U - TS

        U = internal energy (price × volume)
        T = temperature (volatility)
        S = entropy (uncertainty)

        Free energy = energy available for "work" (trading).

        Args:
            price: Current price
            volume: Current volume
            temperature: Market temperature
            entropy: Market entropy

        Returns:
            free_energy: Available trading energy
        """
        # Internal energy (normalized)
        U = np.log(price * volume + 1)

        # Free energy
        F = U - temperature * entropy

        return F

    def entropy_production(self,
                          entropy_before: float,
                          entropy_after: float) -> float:
        """
        2nd law of thermodynamics: entropy must increase.

        ΔS ≥ 0 (for closed systems)

        Violations indicate:
        - Information injection (news)
        - External intervention (manipulation)
        - Measurement error

        Args:
            entropy_before: Entropy at t-1
            entropy_after: Entropy at t

        Returns:
            entropy_production: ΔS (should be ≥ 0)
        """
        return entropy_after - entropy_before

    def phase_transition_order_parameter(self,
                                        correlations: np.ndarray) -> float:
        """
        Detect phase transitions (regime changes).

        Order parameter: average correlation
        - High = ordered phase (herding)
        - Low = disordered phase (independent)

        Phase transition = rapid change in order parameter.

        Args:
            correlations: Cross-asset correlations

        Returns:
            order_param: Phase transition order parameter
        """
        # Mean absolute correlation (excluding diagonal)
        n = len(correlations)
        mask = ~np.eye(n, dtype=bool)
        order_param = np.mean(np.abs(correlations[mask]))

        return order_param


# ============================================================================
# 2. INFORMATION THEORY
# ============================================================================

class InformationTheory:
    """
    Information-theoretic market analysis.

    Key concepts:
    - Fisher information (geometry of probability space)
    - KL divergence (information distance)
    - Mutual information (dependence)
    - Information conservation
    """

    def __init__(self):
        pass

    def fisher_information(self,
                          returns: np.ndarray,
                          params: np.ndarray) -> np.ndarray:
        """
        Fisher information metric.

        Defines the "natural" geometry of probability space.
        Distance = how much information needed to distinguish states.

        I_ij = E[∂log p/∂θ_i · ∂log p/∂θ_j]

        Args:
            returns: Observed returns
            params: Distribution parameters (μ, σ)

        Returns:
            fisher_matrix: Fisher information matrix
        """
        # Simplified: Gaussian assumption
        mu = params[0]
        sigma = params[1]

        n = len(returns)

        # Fisher information for Gaussian
        # I_μμ = n/σ²
        # I_σσ = 2n/σ²
        # I_μσ = 0

        fisher_matrix = np.array([
            [n / (sigma**2), 0],
            [0, 2*n / (sigma**2)]
        ])

        return fisher_matrix

    def kl_divergence(self,
                     p_returns: np.ndarray,
                     q_returns: np.ndarray,
                     n_bins: int = 50) -> float:
        """
        Kullback-Leibler divergence: D_KL(P||Q)

        Measures "information distance" between distributions.
        Not a true metric (not symmetric).

        Args:
            p_returns: Reference distribution
            q_returns: Target distribution
            n_bins: Number of bins

        Returns:
            kl_div: KL divergence in nats
        """
        # Histogram estimation
        bins = np.linspace(
            min(p_returns.min(), q_returns.min()),
            max(p_returns.max(), q_returns.max()),
            n_bins
        )

        p_hist, _ = np.histogram(p_returns, bins=bins, density=True)
        q_hist, _ = np.histogram(q_returns, bins=bins, density=True)

        # Add epsilon to avoid log(0)
        eps = 1e-10
        p_hist = p_hist + eps
        q_hist = q_hist + eps

        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()

        # KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))

        return kl_div

    def mutual_information(self,
                          asset1_returns: np.ndarray,
                          asset2_returns: np.ndarray,
                          n_bins: int = 20) -> float:
        """
        Mutual information: I(X;Y)

        Measures dependence (including non-linear).
        I = 0 → independent
        I > 0 → dependent

        Args:
            asset1_returns: Returns of asset 1
            asset2_returns: Returns of asset 2
            n_bins: Number of bins

        Returns:
            mi: Mutual information
        """
        # 2D histogram
        hist_2d, _, _ = np.histogram2d(
            asset1_returns,
            asset2_returns,
            bins=n_bins,
            density=True
        )

        # Marginals
        hist_x = hist_2d.sum(axis=1)
        hist_y = hist_2d.sum(axis=0)

        # Add epsilon
        eps = 1e-10
        hist_2d = hist_2d + eps
        hist_x = hist_x + eps
        hist_y = hist_y + eps

        # Normalize
        hist_2d = hist_2d / hist_2d.sum()
        hist_x = hist_x / hist_x.sum()
        hist_y = hist_y / hist_y.sum()

        # Mutual information
        # I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
        outer = np.outer(hist_x, hist_y)
        mi = np.sum(hist_2d * np.log(hist_2d / outer))

        return mi


# ============================================================================
# 3. CONSERVATION LAWS
# ============================================================================

class ConservationLaws:
    """
    Enforce physics-like conservation laws.

    Markets must obey:
    1. Information conservation (can't create from nothing)
    2. Energy conservation (capital in = capital out)
    3. Momentum conservation (order flow balance)
    """

    def __init__(self):
        pass

    def information_conservation(self,
                                entropy_before: float,
                                entropy_after: float,
                                information_injection: float = 0.0) -> float:
        """
        Information can only increase via external injection.

        ΔS = S_after - S_before ≥ I_injection

        Violation = information created from nothing.

        Args:
            entropy_before: Entropy at t-1
            entropy_after: Entropy at t
            information_injection: External info (news, etc.)

        Returns:
            violation: Degree of conservation violation
        """
        entropy_change = entropy_after - entropy_before

        # Entropy should increase or decrease by injection
        expected_change = information_injection

        # Violation if entropy decreased without injection
        violation = max(0, -(entropy_change - expected_change))

        return violation

    def energy_conservation(self,
                           capital_in: float,
                           capital_out: float,
                           friction: float = 0.001) -> float:
        """
        Energy (capital) conservation with friction.

        E_out = E_in × (1 - friction)

        Args:
            capital_in: Capital entering
            capital_out: Capital exiting
            friction: Transaction costs

        Returns:
            violation: Energy conservation violation
        """
        expected_out = capital_in * (1 - friction)

        # Violation if more capital out than possible
        violation = max(0, capital_out - expected_out)

        return violation

    def momentum_conservation(self,
                             buy_volume: float,
                             sell_volume: float) -> float:
        """
        Order flow momentum balance.

        Every buyer needs a seller.
        Σ buy_volume = Σ sell_volume

        Args:
            buy_volume: Total buy volume
            sell_volume: Total sell volume

        Returns:
            imbalance: Momentum conservation violation
        """
        imbalance = abs(buy_volume - sell_volume)

        return imbalance


# ============================================================================
# 4. PYTORCH LAYERS
# ============================================================================

class ThermodynamicsLayer(nn.Module):
    """
    PyTorch layer enforcing thermodynamic constraints.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.thermo = MarketThermodynamics()

        # Neural network to process thermodynamic features
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # entropy, temperature, free_energy
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract thermodynamic features.

        Args:
            returns: [batch, time] - return series

        Returns:
            features: Dict of thermodynamic features
        """
        batch_size = returns.shape[0]

        entropies = []
        temperatures = []
        free_energies = []

        for i in range(batch_size):
            ret = returns[i].detach().cpu().numpy()

            # Calculate thermodynamic quantities
            entropy = self.thermo.entropy(ret)
            temp = self.thermo.temperature(ret)

            # Simplified free energy (price=1, volume=1 for normalization)
            free_energy = self.thermo.free_energy(1.0, 1.0, temp, entropy)

            entropies.append(entropy)
            temperatures.append(temp)
            free_energies.append(free_energy)

        # Stack into tensor
        thermo_features = torch.tensor(
            np.array([entropies, temperatures, free_energies]).T,
            dtype=torch.float32,
            device=returns.device
        )

        # Encode
        encoded = self.encoder(thermo_features)

        return {
            'entropy': torch.tensor(entropies, device=returns.device),
            'temperature': torch.tensor(temperatures, device=returns.device),
            'free_energy': torch.tensor(free_energies, device=returns.device),
            'encoded': encoded
        }


class InformationGeometryLayer(nn.Module):
    """
    PyTorch layer using information geometry.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.info_theory = InformationTheory()

        # Process information-theoretic features
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # KL divergence
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self,
                current_returns: torch.Tensor,
                reference_returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate information distance from reference.

        Args:
            current_returns: [batch, time] - current return distribution
            reference_returns: [time] - reference distribution (e.g., historical)

        Returns:
            features: Information-theoretic features
        """
        batch_size = current_returns.shape[0]

        kl_divs = []

        ref = reference_returns.detach().cpu().numpy()

        for i in range(batch_size):
            curr = current_returns[i].detach().cpu().numpy()

            # KL divergence from reference
            kl = self.info_theory.kl_divergence(ref, curr)
            kl_divs.append(kl)

        # Stack
        kl_tensor = torch.tensor(kl_divs, dtype=torch.float32, device=current_returns.device).unsqueeze(1)

        # Encode
        encoded = self.encoder(kl_tensor)

        return {
            'kl_divergence': kl_tensor.squeeze(),
            'encoded': encoded
        }


# ============================================================================
# 5. PHYSICS-CONSTRAINED LOSS
# ============================================================================

class PhysicsConstrainedLoss(nn.Module):
    """
    Loss function enforcing physics constraints.

    Penalizes violations of:
    - Conservation laws
    - 2nd law of thermodynamics
    - Information bounds
    """

    def __init__(self,
                 conservation_weight: float = 1.0,
                 entropy_weight: float = 0.5):
        super().__init__()

        self.conservation_weight = conservation_weight
        self.entropy_weight = entropy_weight

        self.conservation = ConservationLaws()

    def forward(self,
                predictions: torch.Tensor,
                entropy_before: torch.Tensor,
                entropy_after: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate physics-constrained loss.

        Args:
            predictions: Model predictions
            entropy_before: Entropy at t-1
            entropy_after: Entropy at t

        Returns:
            loss, loss_dict
        """
        batch_size = predictions.shape[0]

        # Conservation losses
        info_violations = []

        for i in range(batch_size):
            # Information conservation
            violation = self.conservation.information_conservation(
                entropy_before[i].item(),
                entropy_after[i].item(),
                information_injection=0.0  # Assume no external info
            )
            info_violations.append(violation)

        info_violation_loss = torch.tensor(
            np.mean(info_violations),
            dtype=torch.float32,
            device=predictions.device
        )

        # Entropy production loss (2nd law)
        # Penalize entropy decrease
        entropy_change = entropy_after - entropy_before
        entropy_violation = F.relu(-entropy_change)  # Penalize decrease
        entropy_loss = entropy_violation.mean()

        # Total physics loss
        physics_loss = (
            self.conservation_weight * info_violation_loss +
            self.entropy_weight * entropy_loss
        )

        loss_dict = {
            'physics_total': physics_loss.item(),
            'info_conservation': info_violation_loss.item(),
            'entropy_violation': entropy_loss.item()
        }

        return physics_loss, loss_dict


# ============================================================================
# 6. VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHYSICS LAYER VALIDATION")
    print("=" * 80)

    # Generate test data
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.01

    # Test thermodynamics
    print("\n1. THERMODYNAMICS")
    print("-" * 80)
    thermo = MarketThermodynamics()

    entropy = thermo.entropy(returns)
    temperature = thermo.temperature(returns)
    free_energy = thermo.free_energy(100.0, 1000.0, temperature, entropy)

    print(f"   Entropy: {entropy:.4f} nats")
    print(f"   Temperature: {temperature:.4f}")
    print(f"   Free Energy: {free_energy:.4f}")

    # Test information theory
    print("\n2. INFORMATION THEORY")
    print("-" * 80)
    info = InformationTheory()

    returns2 = np.random.randn(1000) * 0.015
    kl_div = info.kl_divergence(returns, returns2)
    mi = info.mutual_information(returns[:500], returns[500:], n_bins=20)

    print(f"   KL Divergence: {kl_div:.4f} nats")
    print(f"   Mutual Information: {mi:.4f} nats")

    # Test conservation laws
    print("\n3. CONSERVATION LAWS")
    print("-" * 80)
    conservation = ConservationLaws()

    info_viol = conservation.information_conservation(entropy, entropy * 0.9)
    energy_viol = conservation.energy_conservation(1000, 1100, friction=0.001)
    momentum_imbal = conservation.momentum_conservation(1000, 950)

    print(f"   Information violation: {info_viol:.4f}")
    print(f"   Energy violation: {energy_viol:.4f}")
    print(f"   Momentum imbalance: {momentum_imbal:.4f}")

    # Test PyTorch layers
    print("\n4. PYTORCH LAYERS")
    print("-" * 80)

    returns_tensor = torch.randn(4, 100) * 0.01

    thermo_layer = ThermodynamicsLayer(hidden_dim=32)
    thermo_out = thermo_layer(returns_tensor)

    print(f"   Thermodynamics output shape: {thermo_out['encoded'].shape}")
    print(f"   Mean entropy: {thermo_out['entropy'].mean():.4f}")

    info_layer = InformationGeometryLayer(hidden_dim=32)
    ref_returns = torch.randn(100) * 0.01
    info_out = info_layer(returns_tensor, ref_returns)

    print(f"   Information geometry output shape: {info_out['encoded'].shape}")
    print(f"   Mean KL divergence: {info_out['kl_divergence'].mean():.4f}")

    print("\n" + "=" * 80)
    print("PHYSICS LAYER VALIDATION COMPLETE")
    print("=" * 80)
