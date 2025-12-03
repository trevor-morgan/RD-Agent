"""Integration layer bringing Qlib-style orchestration into RD-Agent.

This package hosts the "lab" commands, services, and experimental models that
extend RD-Agent for quant workflows without keeping a separate repository.

Submodules:
    - models: Novel and ensemble models (SymplecticNet, MarketStateNet, etc.)
    - research: RD-Agent adapters, scenario generation, Monte Carlo simulation
    - data: Data collection (Alpaca) and format conversion utilities
    - utils: Data extraction and validation helpers

Example:
    >>> from rdagent_lab.models.novel import SymplecticNet, MarketStateNet
    >>> from rdagent_lab.models.ensemble import PerformanceTracker, WeightingMethod
    >>> from rdagent_lab.research import MonteCarloSimulator
    >>> from rdagent_lab.data import AlpacaCollector
"""

__all__ = [
    "cli",
    "models",
    "research",
    "data",
    "utils",
]
