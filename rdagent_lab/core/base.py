"""Base classes for models and strategies used in lab workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel


class ModelConfig(PydanticBaseModel):
    """Configuration for a model."""

    name: str
    version: str = "0.1.0"
    params: dict[str, Any] = {}


class BaseModel(ABC):
    """Base class for all ML models in the lab."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig(name=self.__class__.__name__)
        self._fitted = False

    @abstractmethod
    def fit(self, dataset: Any) -> "BaseModel":
        """Train the model on the given dataset."""

    @abstractmethod
    def predict(self, dataset: Any) -> pd.Series:
        """Generate predictions for the given dataset."""

    def save(self, path: str) -> None:
        """Persist the model."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a persisted model."""
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._fitted


class StrategyConfig(PydanticBaseModel):
    """Configuration for a strategy."""

    name: str
    params: dict[str, Any] = {}


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig(name=self.__class__.__name__)

    @abstractmethod
    def generate_signals(self, predictions: pd.Series, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from model predictions."""

    @abstractmethod
    def generate_orders(self, signals: pd.DataFrame, portfolio: Any) -> list[dict[str, Any]]:
        """Convert signals to executable orders."""
