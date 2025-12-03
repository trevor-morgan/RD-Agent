"""Model registry and novel architectures.

Submodules:
    - traditional: LightGBM and other tree-based models
    - deep: Transformer and other deep learning models
    - novel: Physics-informed models (SymplecticNet, MarketStateNet)
    - ensemble: Adaptive ensemble utilities and performance tracking

Example:
    >>> from rdagent_lab.models.novel import SymplecticNet, MarketStateNet
    >>> from rdagent_lab.models.ensemble import PerformanceTracker, WeightingMethod
"""

from rdagent_lab.models.traditional import lgbm  # noqa: F401
from rdagent_lab.models.deep import transformer  # noqa: F401

# Novel models
from rdagent_lab.models.novel import (
    SymplecticNet,
    MarketStateNet,
    FractionalDifferencer,
    SymplecticAttention,
    HamiltonianBlock,
    HolographicMemory,
)

# Ensemble utilities
from rdagent_lab.models.ensemble import (
    WeightingMethod,
    PerformanceTracker,
    DiversityRegularizer,
)

__all__ = [
    # Traditional
    "lgbm",
    # Deep
    "transformer",
    # Novel
    "SymplecticNet",
    "MarketStateNet",
    "FractionalDifferencer",
    "SymplecticAttention",
    "HamiltonianBlock",
    "HolographicMemory",
    # Ensemble
    "WeightingMethod",
    "PerformanceTracker",
    "DiversityRegularizer",
]
