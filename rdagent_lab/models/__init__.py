"""Model registry bootstrap."""

from rdagent_lab.models.traditional import lgbm  # noqa: F401
from rdagent_lab.models.deep import transformer  # noqa: F401

__all__ = ["lgbm", "transformer"]
