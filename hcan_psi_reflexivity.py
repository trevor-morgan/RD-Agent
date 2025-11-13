"""
HCAN-Ψ (Psi): Level 5 - Reflexivity Layer

This module implements reflexivity - how models affect markets, creating strange loops.
Markets are self-referential systems where predictions change reality.

Components:
1. Market Impact Models - Price impact from trading
2. Soros Reflexivity - Feedback loops between beliefs and reality
3. Strange Loops - Self-referential dynamics (Hofstadter)
4. Model-Aware Trading - Accounting for model usage by others
5. Quantum-Like Measurement Effects - Observation changes the system

Reference:
- Soros (2013), "The Alchemy of Finance"
- Hofstadter (1979), "Gödel, Escher, Bach"
- Farmer & Lillo (2004), "Theory of Price Impact"

Author: RD-Agent Research Team
Date: 2025-11-13
Level: 5 (Meta-dynamics + Physics + Psychology + Reflexivity)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize


class MarketImpactModel:
    """
    Price impact from trading activity.

    Models:
    - Permanent impact: Δp ∝ Q (order size)
    - Temporary impact: Exponential decay
    - Square-root law: Impact ∝ √Q (Almgren-Chriss)

    Reference: Almgren & Chriss (2000), Farmer & Lillo (2004)
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.3,
        decay_rate: float = 0.5,
        market_depth: float = 1000.0
    ):
        """
        Args:
            permanent_impact_coef: Permanent price impact coefficient
            temporary_impact_coef: Temporary impact coefficient
            decay_rate: Decay rate for temporary impact
            market_depth: Market liquidity depth
        """
        self.gamma = permanent_impact_coef
        self.eta = temporary_impact_coef
        self.kappa = decay_rate
        self.market_depth = market_depth

    def permanent_impact(self, order_size: float, liquidity: float = None) -> float:
        """
        Permanent price impact.

        Δp_perm = γ · Q / L

        Args:
            order_size: Signed order size (positive=buy, negative=sell)
            liquidity: Market liquidity (defaults to market_depth)

        Returns:
            Permanent price change
        """
        if liquidity is None:
            liquidity = self.market_depth

        impact = self.gamma * order_size / liquidity
        return impact

    def temporary_impact(self, order_size: float, liquidity: float = None) -> float:
        """
        Temporary price impact (square-root law).

        Δp_temp = η · sign(Q) · √|Q| / √L

        Args:
            order_size: Signed order size
            liquidity: Market liquidity

        Returns:
            Temporary price change
        """
        if liquidity is None:
            liquidity = self.market_depth

        sign = np.sign(order_size)
        impact = self.eta * sign * np.sqrt(np.abs(order_size)) / np.sqrt(liquidity)
        return impact

    def total_impact(self, order_size: float, liquidity: float = None) -> float:
        """Total impact = permanent + temporary."""
        perm = self.permanent_impact(order_size, liquidity)
        temp = self.temporary_impact(order_size, liquidity)
        return perm + temp

    def impact_decay(self, initial_impact: float, time_elapsed: float) -> float:
        """
        Temporary impact decays exponentially.

        I(t) = I₀ · exp(-κt)

        Args:
            initial_impact: Initial temporary impact
            time_elapsed: Time since order

        Returns:
            Current impact
        """
        return initial_impact * np.exp(-self.kappa * time_elapsed)

    def optimal_execution(
        self,
        total_order_size: float,
        time_horizon: int,
        risk_aversion: float = 0.5
    ) -> np.ndarray:
        """
        Optimal order splitting to minimize impact + risk.

        Almgren-Chriss model.

        Args:
            total_order_size: Total shares to trade
            time_horizon: Number of time steps
            risk_aversion: Risk aversion parameter λ

        Returns:
            Optimal order schedule [time_horizon]
        """
        # Simplified linear strategy
        # Full Almgren-Chriss requires solving differential equations

        # Linear TWAP-like schedule
        schedule = np.linspace(0, total_order_size, time_horizon + 1)
        order_sizes = np.diff(schedule)

        return order_sizes


class SorosReflexivity:
    """
    Soros reflexivity: Feedback between cognition and reality.

    Cycle:
    1. Market participants form beliefs
    2. Beliefs influence actions (buying/selling)
    3. Actions change prices
    4. Prices update beliefs
    5. Repeat → Boom/bust cycles

    Reference: Soros (2013), "The Alchemy of Finance"
    """

    def __init__(
        self,
        belief_update_rate: float = 0.3,
        price_sensitivity: float = 0.5,
        mean_reversion_strength: float = 0.1
    ):
        """
        Args:
            belief_update_rate: How quickly beliefs update from prices
            price_sensitivity: How much prices respond to beliefs
            mean_reversion_strength: Fundamental value pull
        """
        self.alpha = belief_update_rate
        self.beta = price_sensitivity
        self.gamma = mean_reversion_strength

        # State
        self.belief = 0.0  # Market sentiment (-1=bearish, +1=bullish)
        self.price = 100.0
        self.fundamental_value = 100.0

    def update(self, external_shock: float = 0.0) -> Tuple[float, float]:
        """
        Update reflexive dynamics.

        Equations:
        - dP/dt = β·Belief - γ·(P - V) + shock
        - dBelief/dt = α·(dP/dt)

        Args:
            external_shock: External price shock

        Returns:
            (new_price, new_belief)
        """
        # Price responds to beliefs and fundamentals
        price_change = (
            self.beta * self.belief -
            self.gamma * (self.price - self.fundamental_value) +
            external_shock
        )

        self.price += price_change

        # Beliefs update from price momentum
        self.belief += self.alpha * price_change
        self.belief = np.clip(self.belief, -1, 1)

        return self.price, self.belief

    def detect_regime(self) -> str:
        """
        Detect market regime.

        Returns:
            'boom', 'bust', or 'equilibrium'
        """
        deviation = abs(self.price - self.fundamental_value)
        belief_strength = abs(self.belief)

        if deviation > 10 and belief_strength > 0.6:
            if self.belief > 0:
                return 'boom'
            else:
                return 'bust'
        else:
            return 'equilibrium'

    def simulate_cycle(self, n_steps: int = 100, shock_prob: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Simulate reflexive boom-bust cycle.

        Args:
            n_steps: Simulation length
            shock_prob: Probability of external shock per step

        Returns:
            Dictionary with price, belief, regime histories
        """
        prices = np.zeros(n_steps)
        beliefs = np.zeros(n_steps)
        regimes = []

        for t in range(n_steps):
            # Random shock
            shock = np.random.randn() * 2 if np.random.rand() < shock_prob else 0.0

            # Update
            price, belief = self.update(shock)

            # Record
            prices[t] = price
            beliefs[t] = belief
            regimes.append(self.detect_regime())

        return {
            'prices': prices,
            'beliefs': beliefs,
            'regimes': regimes
        }


class StrangeLoops:
    """
    Strange loops: Self-referential hierarchies (Hofstadter).

    Markets create strange loops:
    - Models predict prices
    - Traders use models
    - Model usage changes prices
    - Changed prices invalidate models
    - Back to step 1...

    This is a Gödelian incompleteness in markets.

    Reference: Hofstadter (1979), "Gödel, Escher, Bach"
    """

    def __init__(self, n_levels: int = 3):
        """
        Args:
            n_levels: Number of meta-levels
                Level 0: Reality (actual prices)
                Level 1: Models (price predictions)
                Level 2: Meta-models (models of models)
                Level 3+: Meta-meta-models...
        """
        self.n_levels = n_levels

        # Each level's state
        self.states = [0.0] * n_levels

    def upward_causation(self, level: int, lower_state: float) -> float:
        """
        Lower level affects higher level.

        Level i → Level i+1

        Args:
            level: Current level
            lower_state: State from level below

        Returns:
            Influence on current level
        """
        # Higher levels abstract lower levels
        return np.tanh(lower_state)

    def downward_causation(self, level: int, higher_state: float) -> float:
        """
        Higher level affects lower level.

        Level i+1 → Level i

        This creates the strange loop!

        Args:
            level: Current level
            higher_state: State from level above

        Returns:
            Influence on current level
        """
        # Higher level predictions constrain reality
        return 0.5 * higher_state

    def update(self) -> List[float]:
        """
        Update all levels simultaneously (strange loop dynamics).

        Returns:
            New states [n_levels]
        """
        new_states = [0.0] * self.n_levels

        for level in range(self.n_levels):
            # Upward influence (from below)
            if level > 0:
                upward = self.upward_causation(level, self.states[level - 1])
            else:
                upward = 0.0

            # Downward influence (from above)
            if level < self.n_levels - 1:
                downward = self.downward_causation(level, self.states[level + 1])
            else:
                downward = 0.0

            # Update: mix of up/down influences + self
            new_states[level] = (
                0.5 * self.states[level] +
                0.3 * upward +
                0.2 * downward +
                np.random.randn() * 0.05
            )

        self.states = new_states
        return self.states

    def is_fixed_point(self, tolerance: float = 0.01) -> bool:
        """Check if system reached fixed point (stable strange loop)."""
        old_states = self.states.copy()
        new_states = self.update()

        changes = [abs(new - old) for new, old in zip(new_states, old_states)]
        return all(change < tolerance for change in changes)


class ModelAwareTrading:
    """
    Trading strategy that accounts for other models' presence.

    If many traders use similar models → predict model-driven price moves.

    This is "meta-gaming" the market.
    """

    def __init__(self, model_adoption_rate: float = 0.3):
        """
        Args:
            model_adoption_rate: Fraction of market using similar models
        """
        self.adoption_rate = model_adoption_rate

    def predict_model_driven_move(
        self,
        signal: float,
        market_volume: float,
        model_threshold: float = 0.5
    ) -> float:
        """
        Predict price move from collective model actions.

        If signal > threshold AND many use models → large correlated move.

        Args:
            signal: Model's raw signal
            market_volume: Total market volume
            model_threshold: Signal threshold for model actions

        Returns:
            Expected model-driven price move
        """
        # Will models act?
        if abs(signal) > model_threshold:
            # Fraction of volume from models
            model_volume = market_volume * self.adoption_rate

            # Impact proportional to model volume
            sign = np.sign(signal)
            impact = sign * np.sqrt(model_volume) * 0.01

            return impact
        else:
            return 0.0

    def frontrun_strategy(self, signal: float, threshold: float = 0.5) -> float:
        """
        Frontrun other models.

        If signal > threshold:
            - Other models will buy
            - We buy earlier
            - Sell when they arrive

        Args:
            signal: Observed signal
            threshold: Models' action threshold

        Returns:
            Frontrun position size
        """
        if abs(signal) > threshold:
            # Take position before model herd
            position = np.sign(signal) * min(abs(signal), 1.0)
            return position
        else:
            return 0.0


class QuantumMeasurementEffect:
    """
    Observation changes the system (like quantum measurement).

    In markets:
    - Publishing a forecast changes trader behavior
    - High-frequency observation increases volatility
    - Measurement collapses "superposition" of possible states

    This is analogous to quantum mechanics but classical.

    Reference: von Neumann measurement theory
    """

    def __init__(self, measurement_strength: float = 0.2):
        """
        Args:
            measurement_strength: How much measurement disturbs system
        """
        self.strength = measurement_strength

    def measurement_collapse(
        self,
        state_superposition: np.ndarray,
        observation: float
    ) -> np.ndarray:
        """
        Measurement collapses distribution of possible states.

        Before measurement: broad distribution
        After measurement: concentrated near observed value

        Args:
            state_superposition: Probability distribution over states
            observation: Measured value

        Returns:
            Collapsed distribution (shifted toward observation)
        """
        # Shift distribution toward observation
        n = len(state_superposition)
        observation_idx = int((observation + 1) * n / 2)  # Map [-1,1] → [0,n]
        observation_idx = np.clip(observation_idx, 0, n - 1)

        # Create peaked distribution at observation
        observed_dist = np.exp(-0.5 * ((np.arange(n) - observation_idx) / (n * 0.1)) ** 2)
        observed_dist /= observed_dist.sum()

        # Collapse = weighted mix
        collapsed = (
            (1 - self.strength) * state_superposition +
            self.strength * observed_dist
        )
        collapsed /= collapsed.sum()

        return collapsed

    def observation_volatility(self, base_volatility: float, observation_frequency: float) -> float:
        """
        High-frequency observation increases volatility.

        σ_observed = σ_base · (1 + α · freq)

        Args:
            base_volatility: Intrinsic volatility
            observation_frequency: Observations per unit time

        Returns:
            Amplified volatility
        """
        amplification = 1 + self.strength * observation_frequency
        return base_volatility * amplification


# ====================================================================================================
# PYTORCH NEURAL NETWORK LAYERS
# ====================================================================================================


class MarketImpactLayer(nn.Module):
    """
    Neural network layer for market impact estimation.

    Inputs: Order size, liquidity, historical impact
    Outputs: Expected price impact
    """

    def __init__(self, feature_dim: int = 32):
        """
        Args:
            feature_dim: Output feature dimension
        """
        super().__init__()

        # Impact calculator
        self.impact_model = MarketImpactModel()

        # Encoder: impact features → embeddings
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),  # (permanent_impact, temporary_impact, total_impact)
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, order_sizes: torch.Tensor, liquidity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute market impact.

        Args:
            order_sizes: Planned order sizes [batch_size]
            liquidity: Market liquidity [batch_size]

        Returns:
            Dictionary with:
            - permanent_impact: [batch_size]
            - temporary_impact: [batch_size]
            - total_impact: [batch_size]
            - encoded: [batch_size, feature_dim]
        """
        batch_size = order_sizes.shape[0]

        permanent_impacts = []
        temporary_impacts = []
        total_impacts = []

        for i in range(batch_size):
            order_size = order_sizes[i].item()
            liq = liquidity[i].item()

            perm = self.impact_model.permanent_impact(order_size, liq)
            temp = self.impact_model.temporary_impact(order_size, liq)
            total = perm + temp

            permanent_impacts.append(perm)
            temporary_impacts.append(temp)
            total_impacts.append(total)

        # Convert to tensors
        perm_tensor = torch.tensor(permanent_impacts, dtype=torch.float32, device=order_sizes.device).unsqueeze(1)
        temp_tensor = torch.tensor(temporary_impacts, dtype=torch.float32, device=order_sizes.device).unsqueeze(1)
        total_tensor = torch.tensor(total_impacts, dtype=torch.float32, device=order_sizes.device).unsqueeze(1)

        # Encode
        impact_features = torch.cat([perm_tensor, temp_tensor, total_tensor], dim=1)
        encoded = self.encoder(impact_features)

        return {
            'permanent_impact': perm_tensor.squeeze(1),
            'temporary_impact': temp_tensor.squeeze(1),
            'total_impact': total_tensor.squeeze(1),
            'encoded': encoded
        }


class ReflexivityLayer(nn.Module):
    """
    Soros reflexivity modeling layer.

    Tracks belief-price feedback loops.
    """

    def __init__(self, feature_dim: int = 32):
        """
        Args:
            feature_dim: Output feature dimension
        """
        super().__init__()

        # Reflexivity simulator
        self.reflexivity = SorosReflexivity()

        # Encoder: (price_deviation, belief, regime) → features
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),  # (price_deviation, belief)
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # Regime classifier
        self.regime_classifier = nn.Linear(feature_dim, 3)  # boom, bust, equilibrium

    def forward(self, prices: torch.Tensor, fundamental_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Model reflexive dynamics.

        Args:
            prices: Current prices [batch_size]
            fundamental_values: Estimated fundamental values [batch_size]

        Returns:
            Dictionary with:
            - deviation: Price deviation from fundamentals [batch_size]
            - belief: Market belief [-1, 1] [batch_size]
            - regime_probs: (boom, bust, equilibrium) probabilities [batch_size, 3]
            - encoded: [batch_size, feature_dim]
        """
        batch_size = prices.shape[0]

        deviations = []
        beliefs = []

        for i in range(batch_size):
            price = prices[i].item()
            fundamental = fundamental_values[i].item()

            # Update reflexivity state
            self.reflexivity.price = price
            self.reflexivity.fundamental_value = fundamental
            self.reflexivity.update()

            deviation = price - fundamental
            belief = self.reflexivity.belief

            deviations.append(deviation)
            beliefs.append(belief)

        # Convert to tensors
        dev_tensor = torch.tensor(deviations, dtype=torch.float32, device=prices.device).unsqueeze(1)
        belief_tensor = torch.tensor(beliefs, dtype=torch.float32, device=prices.device).unsqueeze(1)

        # Encode
        reflexivity_features = torch.cat([dev_tensor, belief_tensor], dim=1)
        encoded = self.encoder(reflexivity_features)

        # Classify regime
        regime_logits = self.regime_classifier(encoded)
        regime_probs = F.softmax(regime_logits, dim=1)

        return {
            'deviation': dev_tensor.squeeze(1),
            'belief': belief_tensor.squeeze(1),
            'regime_probs': regime_probs,
            'encoded': encoded
        }


class StrangeLoopLayer(nn.Module):
    """
    Strange loop dynamics layer.

    Models self-referential feedback across meta-levels.
    """

    def __init__(self, n_levels: int = 3, feature_dim: int = 32):
        """
        Args:
            n_levels: Number of meta-levels
            feature_dim: Output feature dimension
        """
        super().__init__()
        self.n_levels = n_levels

        # Strange loop simulator
        self.strange_loop = StrangeLoops(n_levels=n_levels)

        # Encoder: level states → features
        self.encoder = nn.Sequential(
            nn.Linear(n_levels, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, reality_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Propagate state through strange loop.

        Args:
            reality_state: Base reality (level 0) [batch_size]

        Returns:
            Dictionary with:
            - level_states: States at all levels [batch_size, n_levels]
            - encoded: [batch_size, feature_dim]
        """
        batch_size = reality_state.shape[0]

        all_level_states = []

        for i in range(batch_size):
            # Initialize level 0
            self.strange_loop.states[0] = reality_state[i].item()

            # Update strange loop
            level_states = self.strange_loop.update()

            all_level_states.append(level_states)

        # Convert to tensor
        level_tensor = torch.tensor(all_level_states, dtype=torch.float32, device=reality_state.device)

        # Encode
        encoded = self.encoder(level_tensor)

        return {
            'level_states': level_tensor,
            'encoded': encoded
        }


# ====================================================================================================
# VALIDATION
# ====================================================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("REFLEXIVITY LAYER VALIDATION")
    print("=" * 80)

    # 1. Market Impact
    print("\n1. MARKET IMPACT")
    print("-" * 80)
    impact_model = MarketImpactModel()

    order_size = 1000
    perm = impact_model.permanent_impact(order_size)
    temp = impact_model.temporary_impact(order_size)
    total = impact_model.total_impact(order_size)

    print(f"   Order size: {order_size}")
    print(f"   Permanent impact: {perm:.6f}")
    print(f"   Temporary impact: {temp:.6f}")
    print(f"   Total impact: {total:.6f}")

    # Optimal execution
    schedule = impact_model.optimal_execution(total_order_size=10000, time_horizon=10)
    print(f"   Optimal execution schedule: {schedule[:5]} ... (showing first 5)")

    # 2. Soros Reflexivity
    print("\n2. SOROS REFLEXIVITY")
    print("-" * 80)
    reflexivity = SorosReflexivity()

    # Simulate boom-bust cycle
    simulation = reflexivity.simulate_cycle(n_steps=50)
    final_price = simulation['prices'][-1]
    final_belief = simulation['beliefs'][-1]
    final_regime = simulation['regimes'][-1]

    print(f"   Initial price: 100.00")
    print(f"   Final price: {final_price:.2f}")
    print(f"   Final belief: {final_belief:.4f}")
    print(f"   Final regime: {final_regime}")

    # Count regime transitions
    regimes = simulation['regimes']
    boom_pct = sum(1 for r in regimes if r == 'boom') / len(regimes)
    bust_pct = sum(1 for r in regimes if r == 'bust') / len(regimes)
    eq_pct = sum(1 for r in regimes if r == 'equilibrium') / len(regimes)

    print(f"   Time in boom: {boom_pct:.1%}")
    print(f"   Time in bust: {bust_pct:.1%}")
    print(f"   Time in equilibrium: {eq_pct:.1%}")

    # 3. Strange Loops
    print("\n3. STRANGE LOOPS")
    print("-" * 80)
    strange_loop = StrangeLoops(n_levels=3)
    strange_loop.states = [1.0, 0.5, 0.0]  # Initialize

    for t in range(5):
        states = strange_loop.update()
        print(f"   t={t}: Level 0={states[0]:.4f}, Level 1={states[1]:.4f}, Level 2={states[2]:.4f}")

    # 4. Model-Aware Trading
    print("\n4. MODEL-AWARE TRADING")
    print("-" * 80)
    model_aware = ModelAwareTrading(model_adoption_rate=0.3)

    signal = 0.7
    volume = 10000

    model_move = model_aware.predict_model_driven_move(signal, volume)
    frontrun_position = model_aware.frontrun_strategy(signal)

    print(f"   Signal: {signal:.2f}")
    print(f"   Expected model-driven move: {model_move:.6f}")
    print(f"   Frontrun position: {frontrun_position:.4f}")

    # 5. Quantum Measurement Effect
    print("\n5. QUANTUM MEASUREMENT EFFECT")
    print("-" * 80)
    quantum = QuantumMeasurementEffect(measurement_strength=0.2)

    # Superposition before measurement
    n_states = 100
    state_dist = np.ones(n_states) / n_states  # Uniform
    observation = 0.5

    collapsed_dist = quantum.measurement_collapse(state_dist, observation)

    entropy_before = -np.sum(state_dist * np.log(state_dist + 1e-10))
    entropy_after = -np.sum(collapsed_dist * np.log(collapsed_dist + 1e-10))

    print(f"   Entropy before measurement: {entropy_before:.4f}")
    print(f"   Entropy after measurement: {entropy_after:.4f}")
    print(f"   Entropy reduction: {entropy_before - entropy_after:.4f}")

    # Observation amplifies volatility
    base_vol = 0.20
    obs_freq = 100  # High-frequency observation
    amplified_vol = quantum.observation_volatility(base_vol, obs_freq)

    print(f"   Base volatility: {base_vol:.2%}")
    print(f"   Observation frequency: {obs_freq}")
    print(f"   Amplified volatility: {amplified_vol:.2%}")

    # 6. PyTorch Layers
    print("\n6. PYTORCH LAYERS")
    print("-" * 80)

    # Market impact layer
    impact_layer = MarketImpactLayer(feature_dim=32)
    order_sizes = torch.randn(4) * 1000
    liquidity = torch.ones(4) * 1000
    impact_output = impact_layer(order_sizes, liquidity)
    print(f"   Market impact output shape: {impact_output['encoded'].shape}")
    print(f"   Mean total impact: {impact_output['total_impact'].mean():.6f}")

    # Reflexivity layer
    reflexivity_layer = ReflexivityLayer(feature_dim=32)
    prices = torch.tensor([100.0, 105.0, 95.0, 110.0])
    fundamentals = torch.tensor([100.0, 100.0, 100.0, 100.0])
    reflex_output = reflexivity_layer(prices, fundamentals)
    print(f"   Reflexivity output shape: {reflex_output['encoded'].shape}")
    print(f"   Mean deviation: {reflex_output['deviation'].mean():.4f}")
    print(f"   Mean belief: {reflex_output['belief'].mean():.4f}")

    # Strange loop layer
    loop_layer = StrangeLoopLayer(n_levels=3, feature_dim=32)
    reality = torch.randn(4)
    loop_output = loop_layer(reality)
    print(f"   Strange loop output shape: {loop_output['encoded'].shape}")
    print(f"   Level states shape: {loop_output['level_states'].shape}")

    print("\n" + "=" * 80)
    print("REFLEXIVITY LAYER VALIDATION COMPLETE")
    print("=" * 80)
