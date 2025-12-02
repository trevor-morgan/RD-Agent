"""LightGBM model wrapper for Qlib integration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from rdagent_lab.core.base import BaseModel, ModelConfig
from rdagent_lab.core.registry import ModelRegistry
from rdagent_lab.core.exceptions import ModelNotFittedError, ModelTrainingError

_LGBModel = None


def _get_lgb_model():
    """Lazy import to avoid hard dependency unless the extra is installed."""
    global _LGBModel
    if _LGBModel is None:
        from qlib.contrib.model.gbdt import LGBModel

        _LGBModel = LGBModel
    return _LGBModel


class LightGBMConfig(ModelConfig):
    """Configuration for LightGBM model."""

    loss: str = "mse"
    num_leaves: int = 64
    learning_rate: float = 0.05
    n_estimators: int = 500
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = 50
    verbose: int = -1
    fit_intercept: bool = True
    seed: int = 42


@ModelRegistry.register("lgbm")
@ModelRegistry.register("lightgbm")
class LightGBMModel(BaseModel):
    """LightGBM model with Qlib integration and simple save/load."""

    def __init__(self, config: LightGBMConfig | None = None, **kwargs) -> None:
        if config is None:
            config = LightGBMConfig(name="LightGBM", **kwargs)
        else:
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = LightGBMConfig(**config_dict)
        super().__init__(config)
        self._qlib_model = None

    def _get_lgb_params(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "loss": cfg.loss,
            "num_leaves": cfg.num_leaves,
            "learning_rate": cfg.learning_rate,
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_alpha": cfg.reg_alpha,
            "reg_lambda": cfg.reg_lambda,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "verbose": cfg.verbose,
            "seed": cfg.seed,
        }

    def fit(self, dataset: Any) -> "LightGBMModel":
        try:
            LGBModel = _get_lgb_model()
            params = self._get_lgb_params()
            self._qlib_model = LGBModel(**params)
            self._qlib_model.fit(dataset)
            self._fitted = True
            return self
        except Exception as exc:  # noqa: BLE001
            raise ModelTrainingError(self.config.name, str(exc)) from exc

    def predict(self, dataset: Any) -> pd.Series:
        if not self._fitted or self._qlib_model is None:
            raise ModelNotFittedError(self.config.name)
        return self._qlib_model.predict(dataset)

    def save(self, path: str) -> None:
        if self._qlib_model is None:
            raise ModelNotFittedError(self.config.name)
        import joblib

        joblib.dump({"config": self.config.model_dump(), "qlib_model": self._qlib_model}, path)

    @classmethod
    def load(cls, path: str) -> "LightGBMModel":
        import joblib

        data = joblib.load(path)
        config = LightGBMConfig(**data["config"])
        model = cls(config=config)
        model._qlib_model = data["qlib_model"]
        model._fitted = True
        return model

    def get_feature_importance(self) -> pd.Series | None:
        if self._qlib_model is None:
            return None
        try:
            booster = self._qlib_model.model
            importance = booster.feature_importance(importance_type="gain")
            feature_names = booster.feature_name()
            return pd.Series(importance, index=feature_names).sort_values(ascending=False)
        except Exception:  # noqa: BLE001
            return None
