"""Utility functions for RD-Agent Lab.

Provides data extraction, validation, and other helper functions.

Example:
    >>> from rdagent_lab.utils import extract_features_and_labels
    >>> x_train, y_train = extract_features_and_labels(dataset, "train")
"""

from rdagent_lab.utils.data_extraction import (
    DataExtractionError,
    extract_features_and_labels,
    validate_data_for_training,
)

__all__ = [
    "DataExtractionError",
    "extract_features_and_labels",
    "validate_data_for_training",
]
