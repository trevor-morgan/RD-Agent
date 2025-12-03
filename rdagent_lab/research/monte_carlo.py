"""Monte Carlo simulation for scenario analysis.

Generates multiple future scenarios using a trained model
for risk assessment, stress testing, and portfolio optimization.

Components:
    - StressScenario: Definition of a stress scenario
    - ScenarioResult: Container for simulation results with risk metrics
    - MonteCarloSimulator: Main simulator class

Example:
    >>> import torch
    >>> from rdagent_lab.research.monte_carlo import (
    ...     MonteCarloSimulator,
    ...     StressScenario,
    ...     STANDARD_STRESS_SCENARIOS,
    ... )
    >>>
    >>> # Initialize with trained model
    >>> simulator = MonteCarloSimulator(model, device="cuda")
    >>>
    >>> # Run simulation
    >>> initial_state = torch.randn(1, 158)
    >>> result = simulator.simulate(initial_state, n_scenarios=1000, n_steps=20)
    >>> print(result.summary())
    >>>
    >>> # Stress test
    >>> stress_results = simulator.stress_test(
    ...     initial_state,
    ...     scenarios=STANDARD_STRESS_SCENARIOS,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class StressScenario:
    """Definition of a stress scenario.

    Attributes:
        name: Identifier for the scenario
        description: Human-readable description
        shock_magnitude: Size of shock in standard deviations
        affected_features: Feature indices to shock (None = all)
        duration: Days the shock persists
    """

    name: str
    description: str
    shock_magnitude: float
    affected_features: list[int] | None = None
    duration: int = 5


@dataclass
class ScenarioResult:
    """Result of a Monte Carlo simulation.

    Attributes:
        paths: Raw simulation paths [n_scenarios, n_steps, n_features]
        portfolio_returns: Cumulative returns per scenario
        var_95: Value at Risk (95% confidence)
        var_99: Value at Risk (99% confidence)
        cvar_95: Conditional VaR / Expected Shortfall
        max_drawdown: Maximum drawdown across scenarios
        mean_return: Average cumulative return
        std_return: Standard deviation of returns
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis
        positive_scenarios: Count of positive return scenarios
        negative_scenarios: Count of negative return scenarios
    """

    paths: np.ndarray
    portfolio_returns: np.ndarray | None = None
    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    max_drawdown: float | None = None
    mean_return: float | None = None
    std_return: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    positive_scenarios: int = 0
    negative_scenarios: int = 0

    def summary(self) -> dict[str, Any]:
        """Get summary statistics as dictionary."""
        return {
            "n_scenarios": len(self.paths),
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "max_drawdown": self.max_drawdown,
            "positive_pct": self.positive_scenarios / len(self.paths) * 100 if len(self.paths) > 0 else 0,
        }


class MonteCarloSimulator:
    """Monte Carlo scenario simulator.

    Uses a trained prediction model to generate multiple future
    scenarios via simulation. Supports:
    - Risk assessment (VaR, CVaR)
    - Stress testing with predefined scenarios
    - Model uncertainty estimation via MC Dropout

    Args:
        model: Trained nn.Module for predictions
        device: Computation device ("cpu" or "cuda")

    Example:
        >>> simulator = MonteCarloSimulator(model)
        >>> result = simulator.simulate(
        ...     initial_state=torch.randn(1, 158),
        ...     n_scenarios=1000,
        ...     n_steps=20,
        ... )
        >>> print(f"VaR(95%): {result.var_95:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        """Initialize simulator with trained model."""
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def simulate(
        self,
        initial_state: torch.Tensor,
        n_scenarios: int = 1000,
        n_steps: int = 20,
        noise_scale: float = 1.0,
    ) -> ScenarioResult:
        """Generate Monte Carlo scenarios.

        Args:
            initial_state: Starting state [1, d_feat]
            n_scenarios: Number of scenarios to generate
            n_steps: Number of forward steps per scenario
            noise_scale: Scale factor for noise injection

        Returns:
            ScenarioResult with paths and risk metrics
        """
        initial_state = initial_state.to(self.device)

        with torch.no_grad():
            # Expand to n_scenarios
            states = initial_state.expand(n_scenarios, -1).clone()
            all_paths = [states.cpu().numpy()]

            for _ in range(n_steps):
                # Get model prediction
                pred = self._get_prediction(states)

                # Add noise for stochasticity
                noise = torch.randn_like(states) * noise_scale
                states = states + pred + noise

                all_paths.append(states.cpu().numpy())

        paths = np.stack(all_paths, axis=1)  # [n_scenarios, n_steps+1, d_feat]
        return self._compute_statistics(paths)

    def simulate_with_model_uncertainty(
        self,
        initial_state: torch.Tensor,
        n_scenarios: int = 1000,
        n_steps: int = 20,
        n_model_samples: int = 10,
    ) -> ScenarioResult:
        """Simulate with model uncertainty via MC Dropout.

        Args:
            initial_state: Starting state
            n_scenarios: Total scenarios to generate
            n_steps: Forward steps per scenario
            n_model_samples: Number of dropout samples

        Returns:
            Combined results with model uncertainty
        """
        all_paths = []

        # Enable dropout for uncertainty
        self.model.train()

        for _ in range(n_model_samples):
            result = self.simulate(
                initial_state,
                n_scenarios=n_scenarios // n_model_samples,
                n_steps=n_steps,
            )
            all_paths.append(result.paths)

        self.model.eval()

        # Combine all paths
        combined_paths = np.concatenate(all_paths, axis=0)
        return self._compute_statistics(combined_paths)

    def stress_test(
        self,
        initial_state: torch.Tensor,
        scenarios: list[StressScenario],
        n_simulations_per_scenario: int = 100,
        n_steps: int = 20,
    ) -> dict[str, ScenarioResult]:
        """Run stress tests with predefined scenarios.

        Args:
            initial_state: Starting state
            scenarios: List of stress scenarios to test
            n_simulations_per_scenario: Simulations per scenario
            n_steps: Forward steps per simulation

        Returns:
            Dict mapping scenario name to results
        """
        results = {}

        for scenario in scenarios:
            # Apply shock to initial state
            shocked_state = initial_state.clone()

            if scenario.affected_features is None:
                # Shock all features
                shock = torch.randn_like(shocked_state) * scenario.shock_magnitude
                shocked_state = shocked_state + shock
            else:
                # Shock specific features
                for feat_idx in scenario.affected_features:
                    shock = torch.randn(shocked_state.shape[0], 1, device=self.device)
                    shocked_state[:, feat_idx] += shock.squeeze() * scenario.shock_magnitude

            # Simulate from shocked state
            result = self.simulate(
                shocked_state,
                n_scenarios=n_simulations_per_scenario,
                n_steps=n_steps,
                noise_scale=1.5,  # Higher noise for stress scenarios
            )

            results[scenario.name] = result

        return results

    def _get_prediction(self, states: torch.Tensor) -> torch.Tensor:
        """Get prediction from model, handling various output formats."""
        # Try direct model call
        output = self.model(states)

        # Handle dict output (e.g., MarketStateNet)
        if isinstance(output, dict):
            return output.get("prediction", output.get("latent", torch.zeros_like(states[:, :1])))

        return output

    def _compute_statistics(self, paths: np.ndarray) -> ScenarioResult:
        """Compute risk statistics from simulation paths."""
        n_scenarios = paths.shape[0]

        # Compute returns as first feature (assuming it's the target)
        returns = paths[:, 1:, 0] - paths[:, :-1, 0]  # [n_scenarios, n_steps]
        cumulative_returns = returns.sum(axis=1)  # [n_scenarios]

        # Basic stats
        mean_return = float(cumulative_returns.mean())
        std_return = float(cumulative_returns.std())

        # VaR and CVaR
        sorted_returns = np.sort(cumulative_returns)
        var_95 = float(np.percentile(cumulative_returns, 5))
        var_99 = float(np.percentile(cumulative_returns, 1))
        cvar_95 = float(sorted_returns[sorted_returns <= var_95].mean()) if any(sorted_returns <= var_95) else var_95

        # Max drawdown
        cumsum = np.cumsum(returns, axis=1)
        running_max = np.maximum.accumulate(cumsum, axis=1)
        drawdowns = cumsum - running_max
        max_drawdown = float(drawdowns.min())

        # Higher moments
        skewness = float(((cumulative_returns - mean_return) ** 3).mean() / (std_return ** 3 + 1e-8))
        kurtosis = float(((cumulative_returns - mean_return) ** 4).mean() / (std_return ** 4 + 1e-8))

        # Scenario counts
        positive_scenarios = int((cumulative_returns > 0).sum())
        negative_scenarios = n_scenarios - positive_scenarios

        return ScenarioResult(
            paths=paths,
            portfolio_returns=cumulative_returns,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            mean_return=mean_return,
            std_return=std_return,
            skewness=skewness,
            kurtosis=kurtosis,
            positive_scenarios=positive_scenarios,
            negative_scenarios=negative_scenarios,
        )

    @staticmethod
    def compute_var(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Compute VaR and CVaR from returns array.

        Args:
            returns: Array of returns
            confidence: Confidence level (0.95 = 95%)

        Returns:
            Tuple of (VaR, CVaR)
        """
        alpha = 1 - confidence
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean() if any(returns <= var) else var
        return float(var), float(cvar)


# Pre-defined stress scenarios
STANDARD_STRESS_SCENARIOS = [
    StressScenario(
        name="market_crash",
        description="Broad market crash (-3 std devs)",
        shock_magnitude=-3.0,
        duration=5,
    ),
    StressScenario(
        name="volatility_spike",
        description="Sudden volatility increase (+2 std devs)",
        shock_magnitude=2.0,
        duration=3,
    ),
    StressScenario(
        name="flash_crash",
        description="Extreme short-term crash (-5 std devs)",
        shock_magnitude=-5.0,
        duration=1,
    ),
    StressScenario(
        name="gradual_decline",
        description="Steady decline (-1 std dev sustained)",
        shock_magnitude=-1.0,
        duration=20,
    ),
]


__all__ = [
    "MonteCarloSimulator",
    "ScenarioResult",
    "StressScenario",
    "STANDARD_STRESS_SCENARIOS",
]
