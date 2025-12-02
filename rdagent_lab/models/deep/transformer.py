"""Transformer model wrapper for Qlib integration."""

from __future__ import annotations

from typing import Any

import pandas as pd

from rdagent_lab.core.base import BaseModel, ModelConfig
from rdagent_lab.core.registry import ModelRegistry
from rdagent_lab.core.exceptions import ModelNotFittedError, ModelTrainingError

_Transformer = None


def _get_transformer_model():
    """Lazy import to avoid loading torch unless needed."""
    global _Transformer
    if _Transformer is None:
        from qlib.contrib.model.pytorch_transformer import Transformer

        _Transformer = Transformer
    return _Transformer


class TransformerConfig(ModelConfig):
    """Configuration for Transformer model."""

    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.0
    activation: str = "gelu"
    d_feat: int = 6
    seq_len: int = 60
    n_epochs: int = 200
    lr: float = 1e-4
    batch_size: int = 2048
    early_stop: int = 20
    loss: str = "mse"
    optimizer: str = "adam"
    reg: float = 1e-3
    GPU: int = 0
    seed: int = 42


@ModelRegistry.register("transformer")
class TransformerModel(BaseModel):
    """Transformer model with Qlib integration."""

    def __init__(self, config: TransformerConfig | None = None, **kwargs) -> None:
        if config is None:
            config = TransformerConfig(name="Transformer", **kwargs)
        else:
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = TransformerConfig(**config_dict)
        super().__init__(config)
        self._qlib_model = None

    def _get_transformer_params(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_layers": cfg.num_layers,
            "dim_feedforward": cfg.dim_feedforward,
            "dropout": cfg.dropout,
            "activation": cfg.activation,
            "d_feat": cfg.d_feat,
            "n_epochs": cfg.n_epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "early_stop": cfg.early_stop,
            "loss": cfg.loss,
            "optimizer": cfg.optimizer,
            "reg": cfg.reg,
            "GPU": cfg.GPU,
            "seed": cfg.seed,
        }

    def fit(self, dataset: Any) -> "TransformerModel":
        try:
            Transformer = _get_transformer_model()
            params = self._get_transformer_params()
            self._qlib_model = Transformer(**params)
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
        import torch

        joblib.dump({"config": self.config.model_dump()}, f"{path}.config")
        if hasattr(self._qlib_model, "model"):
            torch.save(self._qlib_model.model.state_dict(), f"{path}.pt")

    @classmethod
    def load(cls, path: str) -> "TransformerModel":
        import joblib
        import torch

        data = joblib.load(f"{path}.config")
        config = TransformerConfig(**data["config"])
        model = cls(config=config)
        Transformer = _get_transformer_model()
        params = model._get_transformer_params()
        model._qlib_model = Transformer(**params)
        if hasattr(model._qlib_model, "model"):
            state_dict = torch.load(f"{path}.pt", map_location="cpu")
            model._qlib_model.model.load_state_dict(state_dict)
        model._fitted = True
        return model
