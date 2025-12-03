"""Utilities for safely extracting data from Qlib datasets.

Provides robust data extraction with validation and error handling for
different Qlib DatasetH output formats.

Example:
    >>> from rdagent_lab.utils import extract_features_and_labels
    >>> x_train, y_train = extract_features_and_labels(dataset, "train")
    >>> print(f"Shape: {x_train.shape}, {y_train.shape}")
    Shape: (10000, 158), (10000,)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DataExtractionError(Exception):
    """Raised when data extraction from Qlib dataset fails."""

    pass


def extract_features_and_labels(
    dataset: Any,
    segment: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from a Qlib DatasetH.

    Handles various Qlib data formats and validates shapes.

    Args:
        dataset: Qlib DatasetH instance
        segment: Data segment ("train", "valid", "test")

    Returns:
        Tuple of (features, labels) as numpy arrays
        - features: 2D array of shape (n_samples, n_features)
        - labels: 1D array of shape (n_samples,)

    Raises:
        DataExtractionError: If data extraction or validation fails

    Example:
        >>> x_train, y_train = extract_features_and_labels(dataset, "train")
        >>> print(f"Shape: {x_train.shape}, {y_train.shape}")
        Shape: (10000, 158), (10000,)
    """
    try:
        df = dataset.prepare(segment, col_set=["feature", "label"])
    except Exception as e:
        raise DataExtractionError(
            f"Failed to prepare dataset segment '{segment}': {e}"
        ) from e

    if df is None or len(df) == 0:
        raise DataExtractionError(
            f"Empty data from dataset segment '{segment}'. "
            "Check that your date ranges have data and data provider is initialized."
        )

    # Extract features
    features = _extract_column_as_array(df, "feature")

    # Extract labels
    labels = _extract_column_as_array(df, "label")
    labels = labels.squeeze()  # Flatten to 1D

    # Validate shapes
    if features.ndim != 2:
        raise DataExtractionError(
            f"Expected 2D feature array, got {features.ndim}D with shape {features.shape}. "
            "Check your data handler configuration."
        )

    if labels.ndim != 1:
        raise DataExtractionError(
            f"Expected 1D label array, got {labels.ndim}D with shape {labels.shape}. "
            "Check your label configuration."
        )

    if len(features) != len(labels):
        raise DataExtractionError(
            f"Feature/label length mismatch: {len(features)} features vs {len(labels)} labels"
        )

    logger.debug(
        f"Extracted {segment} data: {len(features):,} samples, "
        f"{features.shape[1]} features"
    )

    return features, labels


def _extract_column_as_array(df: pd.DataFrame, column: str) -> np.ndarray:
    """Extract a column from a DataFrame as a properly shaped numpy array.

    Handles various DataFrame structures from Qlib:
    - Standard DataFrames with MultiIndex
    - DataFrames with nested object arrays
    - DataFrames with regular numeric columns

    Args:
        df: Pandas DataFrame from DatasetH.prepare()
        column: Column name to extract ("feature" or "label")

    Returns:
        Numpy array with proper shape

    Raises:
        DataExtractionError: If column extraction fails
    """
    if column not in df.columns.get_level_values(0):
        raise DataExtractionError(
            f"Column '{column}' not found in DataFrame. "
            f"Available columns: {list(df.columns.get_level_values(0).unique())}"
        )

    col_data = df[column]

    # If it's a DataFrame (multiple sub-columns), convert to values
    if isinstance(col_data, pd.DataFrame):
        values = col_data.values
        # Check for object dtype (nested arrays)
        if values.dtype == object:
            values = _convert_object_array(values)
        return values.astype(np.float32)

    # If it's a Series
    if isinstance(col_data, pd.Series):
        values = col_data.values
        if values.dtype == object:
            values = _convert_object_array(values)
        return values.astype(np.float32)

    raise DataExtractionError(
        f"Unexpected type for column '{column}': {type(col_data)}"
    )


def _convert_object_array(arr: np.ndarray) -> np.ndarray:
    """Convert object dtype array to numeric array.

    Handles cases where Qlib stores arrays as object dtype
    (e.g., each row is a numpy array stored as an object).

    Args:
        arr: Numpy array with object dtype

    Returns:
        Properly stacked numeric array
    """
    if arr.ndim == 1:
        # 1D array of objects - each element might be an array
        if len(arr) > 0 and isinstance(arr[0], (np.ndarray, list)):
            return np.vstack(arr)
        # Just a 1D array of scalars stored as objects
        return arr.astype(np.float32)

    if arr.ndim == 2:
        # 2D array of objects - try to convert directly
        try:
            return arr.astype(np.float32)
        except (ValueError, TypeError):
            # Fallback: flatten and reshape
            flat = np.array([x for row in arr for x in row], dtype=np.float32)
            return flat.reshape(arr.shape)

    # Higher dimensions - attempt direct conversion
    return arr.astype(np.float32)


def validate_data_for_training(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int = 100,
) -> None:
    """Validate data is suitable for training.

    Args:
        x: Feature array (2D)
        y: Label array (1D)
        min_samples: Minimum required samples

    Raises:
        DataExtractionError: If validation fails
    """
    if x.ndim != 2:
        raise DataExtractionError(f"Features must be 2D, got {x.ndim}D")

    if y.ndim != 1:
        raise DataExtractionError(f"Labels must be 1D, got {y.ndim}D")

    if len(x) != len(y):
        raise DataExtractionError(
            f"Feature/label count mismatch: {len(x)} vs {len(y)}"
        )

    if len(x) < min_samples:
        raise DataExtractionError(
            f"Insufficient samples: {len(x)} < {min_samples}. "
            "Check your date ranges or data availability."
        )

    # Check for all-NaN columns - warn but don't fail
    nan_cols = np.all(np.isnan(x), axis=0)
    if np.any(nan_cols):
        n_nan_cols = int(np.sum(nan_cols))
        logger.warning(
            f"Found {n_nan_cols} all-NaN feature columns. "
            "These will be filled with 0. Check your data handler or date ranges."
        )

    # Check for all-NaN labels
    if np.all(np.isnan(y)):
        raise DataExtractionError("All labels are NaN. Check your label configuration.")

    # Warn about high NaN ratio
    nan_ratio = np.mean(np.isnan(x))
    if nan_ratio > 0.5:
        logger.warning(
            f"High NaN ratio in features: {nan_ratio:.1%}. "
            "Consider checking data quality."
        )


__all__ = [
    "DataExtractionError",
    "extract_features_and_labels",
    "validate_data_for_training",
]
