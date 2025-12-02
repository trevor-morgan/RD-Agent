"""Tests for custom exceptions."""

from __future__ import annotations

import pytest

from rdagent_lab.core.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    ModelNotFittedError,
    ModelTrainingError,
)


# --- ConfigurationError tests ---


def test_configuration_error_stores_field_and_message() -> None:
    """Should store field and message attributes."""
    exc = ConfigurationError("model_type", "Unknown model")
    assert exc.field == "model_type"
    assert exc.message == "Unknown model"


def test_configuration_error_str_contains_field() -> None:
    """Should include field in string representation."""
    exc = ConfigurationError("model_type", "Unknown model")
    assert "model_type" in str(exc)
    assert "Unknown model" in str(exc)


def test_configuration_error_is_raiseable() -> None:
    """Should be raiseable as exception."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("test_field", "test message")
    assert exc_info.value.field == "test_field"


# --- DataNotFoundError tests ---


def test_data_not_found_error_stores_name_and_path() -> None:
    """Should store name and path attributes."""
    exc = DataNotFoundError("qlib_data", "/path/to/data")
    assert exc.name == "qlib_data"
    assert exc.path == "/path/to/data"


def test_data_not_found_error_str_contains_name_and_path() -> None:
    """Should include name and path in string representation."""
    exc = DataNotFoundError("qlib_data", "/path/to/data")
    assert "qlib_data" in str(exc)
    assert "/path/to/data" in str(exc)


def test_data_not_found_error_is_raiseable() -> None:
    """Should be raiseable as exception."""
    with pytest.raises(DataNotFoundError) as exc_info:
        raise DataNotFoundError("test_data", "/test/path")
    assert exc_info.value.name == "test_data"


# --- ModelTrainingError tests ---


def test_model_training_error_stores_model_and_reason() -> None:
    """Should store model and reason attributes."""
    exc = ModelTrainingError("LightGBM", "Out of memory")
    assert exc.model == "LightGBM"
    assert exc.reason == "Out of memory"


def test_model_training_error_str_contains_model_and_reason() -> None:
    """Should include model and reason in string representation."""
    exc = ModelTrainingError("LightGBM", "Out of memory")
    assert "LightGBM" in str(exc)
    assert "Out of memory" in str(exc)
    assert "failed" in str(exc).lower()


def test_model_training_error_is_raiseable() -> None:
    """Should be raiseable as exception."""
    with pytest.raises(ModelTrainingError) as exc_info:
        raise ModelTrainingError("TestModel", "test reason")
    assert exc_info.value.model == "TestModel"


# --- ModelNotFittedError tests ---


def test_model_not_fitted_error_stores_model() -> None:
    """Should store model attribute."""
    exc = ModelNotFittedError("Transformer")
    assert exc.model == "Transformer"


def test_model_not_fitted_error_str_contains_model_and_fitted() -> None:
    """Should include model name and fitted message."""
    exc = ModelNotFittedError("Transformer")
    assert "Transformer" in str(exc)
    assert "fitted" in str(exc).lower()


def test_model_not_fitted_error_is_raiseable() -> None:
    """Should be raiseable as exception."""
    with pytest.raises(ModelNotFittedError) as exc_info:
        raise ModelNotFittedError("TestModel")
    assert exc_info.value.model == "TestModel"
