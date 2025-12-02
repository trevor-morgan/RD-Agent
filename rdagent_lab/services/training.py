"""Training service for orchestrating Qlib model training workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from rdagent_lab.core.base import BaseModel
from rdagent_lab.core.registry import ModelRegistry
from rdagent_lab.core.exceptions import (
    ModelTrainingError,
    ConfigurationError,
    DataNotFoundError,
)


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    model_type: str = "lgbm"
    model_params: dict[str, Any] = field(default_factory=dict)
    instruments: str = "csi300"
    start_time: str = "2018-01-01"
    end_time: str = "2020-08-01"
    fit_start_time: str = "2018-01-01"
    fit_end_time: str = "2019-12-31"
    feature_config: str = "Alpha158"
    label_config: str = "Ref($close, -2) / Ref($close, -1) - 1"
    experiment_name: str = "rdagent_lab"
    save_path: str | None = None
    qlib_provider_uri: str = "~/.qlib/qlib_data/cn_data"
    qlib_region: str = "cn"


@dataclass
class TrainingResult:
    """Result of a training run."""

    model: BaseModel
    metrics: dict[str, float]
    predictions: pd.Series | None = None
    feature_importance: pd.Series | None = None
    experiment_id: str | None = None
    save_path: str | None = None


class TrainingService:
    """High-level service for model training using Qlib."""

    def __init__(self, auto_init_qlib: bool = True) -> None:
        self._qlib_initialized = False
        self._auto_init = auto_init_qlib

    def _ensure_qlib_initialized(self, config: TrainingConfig) -> None:
        if self._qlib_initialized:
            return
        try:
            import qlib
            from qlib.config import REG_CN, REG_US

            region = REG_CN if config.qlib_region == "cn" else REG_US
            provider_uri = Path(config.qlib_provider_uri).expanduser()
            if not provider_uri.exists():
                raise DataNotFoundError("qlib_data", str(provider_uri))
            qlib.init(provider_uri=str(provider_uri), region=region)
            self._qlib_initialized = True
            logger.info(f"Qlib initialized with provider: {provider_uri}")
        except ImportError as exc:  # noqa: BLE001
            raise ConfigurationError("qlib", f"Qlib not installed: {exc}") from exc

    def _create_dataset(self, config: TrainingConfig) -> Any:
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP

        if config.feature_config == "Alpha158":
            from qlib.contrib.data.handler import Alpha158

            handler: DataHandlerLP | None = Alpha158(
                instruments=config.instruments,
                start_time=config.start_time,
                end_time=config.end_time,
                fit_start_time=config.fit_start_time,
                fit_end_time=config.fit_end_time,
            )
        elif config.feature_config == "Alpha360":
            from qlib.contrib.data.handler import Alpha360

            handler = Alpha360(
                instruments=config.instruments,
                start_time=config.start_time,
                end_time=config.end_time,
                fit_start_time=config.fit_start_time,
                fit_end_time=config.fit_end_time,
            )
        else:
            raise ConfigurationError("feature_config", f"Unknown feature config: {config.feature_config}")

        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (config.fit_start_time, config.fit_end_time),
                "valid": (config.fit_end_time, config.end_time),
                "test": (config.fit_end_time, config.end_time),
            },
        )
        return dataset

    def _create_model(self, config: TrainingConfig) -> BaseModel:
        from rdagent_lab import models as _  # noqa: F401

        if config.model_type not in ModelRegistry:
            raise ConfigurationError("model_type", f"Unknown model type: {config.model_type}")
        return ModelRegistry.create(config.model_type, **config.model_params)

    def train(self, config: TrainingConfig) -> TrainingResult:
        logger.info(f"Starting training: model={config.model_type}, features={config.feature_config}")
        if self._auto_init:
            self._ensure_qlib_initialized(config)
        dataset = self._create_dataset(config)
        model = self._create_model(config)
        try:
            model.fit(dataset)
        except Exception as exc:  # noqa: BLE001
            raise ModelTrainingError(config.model_type, str(exc)) from exc

        predictions = model.predict(dataset)
        feature_importance = None
        if hasattr(model, "get_feature_importance"):
            feature_importance = getattr(model, "get_feature_importance")()

        metrics = self._calculate_metrics(predictions, dataset)

        save_path = None
        if config.save_path:
            save_path = config.save_path
            model.save(save_path)
            logger.info(f"Model saved to: {save_path}")

        return TrainingResult(
            model=model,
            metrics=metrics,
            predictions=predictions,
            feature_importance=feature_importance,
            experiment_id=config.experiment_name,
            save_path=save_path,
        )

    def _calculate_metrics(self, predictions: pd.Series, dataset: Any) -> dict[str, float]:
        metrics = {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "rank_ic_mean": 0.0, "mse": 0.0, "hit_rate": 0.0}
        if predictions is None or len(predictions) == 0:
            return metrics
        try:
            df_test = dataset.prepare("test", col_set=["label"])
            labels = df_test["label"].squeeze()
            aligned_preds, aligned_labels = predictions.align(labels, join="inner")
            valid_mask = ~(aligned_preds.isna() | aligned_labels.isna())
            aligned_preds = aligned_preds[valid_mask]
            aligned_labels = aligned_labels[valid_mask]
            if len(aligned_preds) == 0:
                return metrics
            ic_per_day = aligned_preds.groupby(level=0).apply(lambda pred: pred.corr(aligned_labels.loc[pred.index]))
            ic_per_day = ic_per_day.dropna()
            if len(ic_per_day) > 0:
                metrics["ic_mean"] = float(ic_per_day.mean())
                metrics["ic_std"] = float(ic_per_day.std())
                if metrics["ic_std"] > 1e-8:
                    metrics["icir"] = metrics["ic_mean"] / metrics["ic_std"]
            rank_ic_per_day = aligned_preds.groupby(level=0).apply(
                lambda pred: pred.rank().corr(aligned_labels.loc[pred.index].rank())
            )
            rank_ic_per_day = rank_ic_per_day.dropna()
            if len(rank_ic_per_day) > 0:
                metrics["rank_ic_mean"] = float(rank_ic_per_day.mean())
            metrics["mse"] = float(((aligned_preds - aligned_labels) ** 2).mean())
            pred_direction = (aligned_preds > 0).astype(int)
            label_direction = (aligned_labels > 0).astype(int)
            metrics["hit_rate"] = float((pred_direction == label_direction).mean())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not calculate metrics: {exc}")
        return metrics

    def train_from_yaml(self, config_path: str | Path) -> TrainingResult:
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError("config_path", f"File not found: {config_path}")
        with open(config_path) as handle:
            workflow_config = yaml.safe_load(handle)

        training_config = TrainingConfig(
            experiment_name=workflow_config.get("experiment_name", "qlib_workflow"),
            qlib_provider_uri=workflow_config.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
            qlib_region=workflow_config.get("region", "cn"),
        )
        if "task" in workflow_config:
            task = workflow_config["task"]
            if "model" in task:
                model_config = task["model"]
                training_config.model_type = model_config.get("class", "lgbm").split(".")[-1].lower()
                training_config.model_params = model_config.get("kwargs", {})
            if "dataset" in task and "kwargs" in task["dataset"]:
                kwargs = task["dataset"]["kwargs"]
                if "segments" in kwargs and "train" in kwargs["segments"]:
                    training_config.fit_start_time = kwargs["segments"]["train"][0]
                    training_config.fit_end_time = kwargs["segments"]["train"][1]
        return self.train(training_config)

    def train_with_qlib_workflow(self, config_path: str | Path) -> dict[str, Any]:
        """Use Qlib's native workflow runner for full compatibility."""
        from qlib.workflow import R
        from qlib.utils import init_instance_by_config
        import yaml

        config_path = Path(config_path)
        with open(config_path) as handle:
            config = yaml.safe_load(handle)
        if "qlib_init" in config:
            import qlib

            qlib.init(**config["qlib_init"])
        with R.start(experiment_name=config.get("experiment_name", "rdagent_lab")):
            model = init_instance_by_config(config["task"]["model"])
            dataset = init_instance_by_config(config["task"]["dataset"])
            model.fit(dataset)
            pred = model.predict(dataset)
            R.save_objects(pred=pred, trained_model=model)
            recorder = R.get_recorder()
            return {
                "model": model,
                "predictions": pred,
                "recorder_id": recorder.id if recorder else None,
                "experiment_name": config.get("experiment_name"),
            }
