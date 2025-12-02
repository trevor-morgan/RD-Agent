"""Tests for TrainingService."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from rdagent_lab.core.exceptions import ConfigurationError, DataNotFoundError
from rdagent_lab.services.training import (
    TrainingConfig,
    TrainingResult,
    TrainingService,
)


# --- TrainingConfig tests ---


def test_training_config_default_values() -> None:
    """Should have sensible defaults."""
    config = TrainingConfig()
    assert config.model_type == "lgbm"
    assert config.instruments == "csi300"
    assert config.feature_config == "Alpha158"
    assert config.qlib_region == "cn"


def test_training_config_custom_values() -> None:
    """Should accept custom values."""
    config = TrainingConfig(
        model_type="transformer",
        instruments="csi500",
        feature_config="Alpha360",
        qlib_region="us",
    )
    assert config.model_type == "transformer"
    assert config.instruments == "csi500"
    assert config.feature_config == "Alpha360"
    assert config.qlib_region == "us"


def test_training_config_model_params_default_empty() -> None:
    """Should default model_params to empty dict."""
    config = TrainingConfig()
    assert config.model_params == {}


def test_training_config_model_params_custom() -> None:
    """Should accept custom model params."""
    config = TrainingConfig(model_params={"n_estimators": 100, "learning_rate": 0.1})
    assert config.model_params["n_estimators"] == 100
    assert config.model_params["learning_rate"] == 0.1


# --- TrainingResult tests ---


def test_training_result_required_fields() -> None:
    """Should require model and metrics."""
    mock_model = Mock()
    result = TrainingResult(
        model=mock_model,
        metrics={"ic_mean": 0.05, "sharpe": 1.2},
    )
    assert result.model is mock_model
    assert result.metrics["ic_mean"] == 0.05


def test_training_result_optional_fields_default_none() -> None:
    """Should default optional fields to None."""
    mock_model = Mock()
    result = TrainingResult(model=mock_model, metrics={})
    assert result.predictions is None
    assert result.feature_importance is None
    assert result.experiment_id is None
    assert result.save_path is None


def test_training_result_optional_fields_custom() -> None:
    """Should accept custom optional fields."""
    mock_model = Mock()
    predictions = pd.Series([0.1, 0.2, 0.3])
    feature_importance = pd.Series({"a": 0.5, "b": 0.3})
    result = TrainingResult(
        model=mock_model,
        metrics={},
        predictions=predictions,
        feature_importance=feature_importance,
        experiment_id="exp_001",
        save_path="/tmp/model.pkl",
    )
    assert result.predictions is predictions
    assert result.feature_importance is feature_importance
    assert result.experiment_id == "exp_001"
    assert result.save_path == "/tmp/model.pkl"


# --- TrainingService tests ---


def test_training_service_init_auto_qlib_default() -> None:
    """Should default to auto-initializing qlib."""
    service = TrainingService()
    assert service._auto_init is True
    assert service._qlib_initialized is False


def test_training_service_init_qlib_disabled() -> None:
    """Should allow disabling auto qlib init."""
    service = TrainingService(auto_init_qlib=False)
    assert service._auto_init is False


def test_training_service_ensure_qlib_data_not_found() -> None:
    """Should raise DataNotFoundError if qlib data missing."""
    with patch("rdagent_lab.services.training.Path") as mock_path:
        mock_path.return_value.expanduser.return_value.exists.return_value = False

        service = TrainingService()
        config = TrainingConfig(qlib_provider_uri="/nonexistent")

        with pytest.raises(DataNotFoundError):
            service._ensure_qlib_initialized(config)


def test_training_service_create_model_unknown_type() -> None:
    """Should raise ConfigurationError for unknown model type."""
    service = TrainingService(auto_init_qlib=False)
    config = TrainingConfig(model_type="unknown_model_xyz")

    with pytest.raises(ConfigurationError, match="Unknown model type"):
        service._create_model(config)


def test_training_service_calculate_metrics_empty_predictions() -> None:
    """Should return zeros for empty predictions."""
    service = TrainingService(auto_init_qlib=False)
    empty_preds = pd.Series([], dtype=float)
    mock_dataset = Mock()

    metrics = service._calculate_metrics(empty_preds, mock_dataset)

    assert metrics["ic_mean"] == 0.0
    assert metrics["icir"] == 0.0
    assert metrics["mse"] == 0.0


def test_training_service_calculate_metrics_none_predictions() -> None:
    """Should return zeros for None predictions."""
    service = TrainingService(auto_init_qlib=False)
    mock_dataset = Mock()

    metrics = service._calculate_metrics(None, mock_dataset)

    assert metrics["ic_mean"] == 0.0


def test_training_service_calculate_metrics_with_valid_data() -> None:
    """Should calculate metrics with valid aligned data."""
    service = TrainingService(auto_init_qlib=False)

    # Create multi-index predictions
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    symbols = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, symbols])
    predictions = pd.Series([0.1, -0.1] * 10, index=index)

    # Mock dataset that returns labels
    mock_dataset = Mock()
    labels_df = pd.DataFrame({"label": [0.05, -0.05] * 10}, index=index)
    mock_dataset.prepare.return_value = labels_df

    metrics = service._calculate_metrics(predictions, mock_dataset)

    # Should have calculated some metrics
    assert "ic_mean" in metrics
    assert "rank_ic_mean" in metrics
    assert "mse" in metrics
    assert "hit_rate" in metrics
