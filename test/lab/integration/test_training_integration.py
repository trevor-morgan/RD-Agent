"""Integration tests for training service with real Qlib data.

These tests are skipped if Qlib data is not available or if the
Qlib API version is incompatible.
Run with: pytest test/lab/integration/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import handle_qlib_api_error


# These tests require Qlib data - skip marker is in conftest.py


@handle_qlib_api_error
def test_training_service_lgbm_smoke(
    initialized_qlib, qlib_data_path: Path, qlib_region: str, short_date_range: dict
) -> None:
    """Should complete LightGBM training without error."""
    from rdagent_lab.services.training import TrainingConfig, TrainingService

    config = TrainingConfig(
        model_type="lgbm",
        instruments="csi300" if qlib_region == "cn" else "sp500",
        feature_config="Alpha158",
        qlib_provider_uri=str(qlib_data_path),
        qlib_region=qlib_region,
        **short_date_range,
    )

    service = TrainingService(auto_init_qlib=False)  # Already initialized by fixture
    result = service.train(config)

    assert result.model is not None
    assert "ic_mean" in result.metrics
    assert result.predictions is not None


@handle_qlib_api_error
def test_training_service_transformer_smoke(
    initialized_qlib, qlib_data_path: Path, qlib_region: str, short_date_range: dict
) -> None:
    """Should complete Transformer training without error."""
    from rdagent_lab.services.training import TrainingConfig, TrainingService

    config = TrainingConfig(
        model_type="transformer",
        instruments="csi300" if qlib_region == "cn" else "sp500",
        feature_config="Alpha158",
        qlib_provider_uri=str(qlib_data_path),
        qlib_region=qlib_region,
        model_params={"d_model": 32, "n_heads": 2, "num_layers": 1},  # Small for speed
        **short_date_range,
    )

    service = TrainingService(auto_init_qlib=False)
    result = service.train(config)

    assert result.model is not None
    assert "ic_mean" in result.metrics


@handle_qlib_api_error
def test_training_service_alpha360_features(
    initialized_qlib, qlib_data_path: Path, qlib_region: str, short_date_range: dict
) -> None:
    """Should work with Alpha360 feature set."""
    from rdagent_lab.services.training import TrainingConfig, TrainingService

    config = TrainingConfig(
        model_type="lgbm",
        instruments="csi300" if qlib_region == "cn" else "sp500",
        feature_config="Alpha360",
        qlib_provider_uri=str(qlib_data_path),
        qlib_region=qlib_region,
        **short_date_range,
    )

    service = TrainingService(auto_init_qlib=False)
    result = service.train(config)

    assert result.model is not None


@handle_qlib_api_error
def test_training_metrics_reasonable_values(
    initialized_qlib, qlib_data_path: Path, qlib_region: str, short_date_range: dict
) -> None:
    """Metrics should be within reasonable ranges."""
    from rdagent_lab.services.training import TrainingConfig, TrainingService

    config = TrainingConfig(
        model_type="lgbm",
        instruments="csi300" if qlib_region == "cn" else "sp500",
        feature_config="Alpha158",
        qlib_provider_uri=str(qlib_data_path),
        qlib_region=qlib_region,
        **short_date_range,
    )

    service = TrainingService(auto_init_qlib=False)
    result = service.train(config)

    # IC should be between -1 and 1
    assert -1 <= result.metrics["ic_mean"] <= 1
    # Hit rate should be between 0 and 1
    assert 0 <= result.metrics["hit_rate"] <= 1
    # MSE should be non-negative
    assert result.metrics["mse"] >= 0


@handle_qlib_api_error
def test_training_with_model_save(
    initialized_qlib, qlib_data_path: Path, qlib_region: str, short_date_range: dict, tmp_path: Path
) -> None:
    """Should save model to specified path."""
    from rdagent_lab.services.training import TrainingConfig, TrainingService

    save_path = str(tmp_path / "model.pkl")

    config = TrainingConfig(
        model_type="lgbm",
        instruments="csi300" if qlib_region == "cn" else "sp500",
        feature_config="Alpha158",
        qlib_provider_uri=str(qlib_data_path),
        qlib_region=qlib_region,
        save_path=save_path,
        **short_date_range,
    )

    service = TrainingService(auto_init_qlib=False)
    result = service.train(config)

    assert result.save_path == save_path
    assert Path(save_path).exists()
