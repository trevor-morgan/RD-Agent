"""Novel neural network architectures for quantitative finance.

This module contains physics-inspired and research-stage neural network
architectures that go beyond standard deep learning approaches.

Available Models:
    - SymplecticNet: Physics-informed network with symplectic attention,
      Hamiltonian dynamics, and holographic memory
    - MarketStateNet: Multi-feature market state model with topology,
      complexity, and activity metrics

Example:
    >>> from rdagent_lab.models.novel import SymplecticNet, MarketStateNet
    >>> model = SymplecticNet(d_feat=158, d_model=64)
"""

from rdagent_lab.models.novel.symplectic import (
    FractionalDifferencer,
    HamiltonianBlock,
    HolographicMemory,
    SymplecticAttention,
    SymplecticNet,
)
from rdagent_lab.models.novel.market_state import (
    TopologyFeatures,
    CompressionComplexity,
    ActivityMetrics,
    QuantilePredictor,
    AnomalyDetector,
    MarketStateNet,
)

__all__ = [
    # Symplectic components
    "FractionalDifferencer",
    "SymplecticAttention",
    "HamiltonianBlock",
    "HolographicMemory",
    "SymplecticNet",
    # Market state components
    "TopologyFeatures",
    "CompressionComplexity",
    "ActivityMetrics",
    "QuantilePredictor",
    "AnomalyDetector",
    "MarketStateNet",
]
