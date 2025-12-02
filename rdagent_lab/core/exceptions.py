"""Shared exception types for lab services."""


class ConfigurationError(Exception):
    """Raised when a configuration value is invalid or missing."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class DataNotFoundError(Exception):
    """Raised when expected data is missing on disk."""

    def __init__(self, name: str, path: str) -> None:
        super().__init__(f"{name} not found at {path}")
        self.name = name
        self.path = path


class ModelTrainingError(Exception):
    """Raised when model training fails."""

    def __init__(self, model: str, reason: str) -> None:
        super().__init__(f"{model} training failed: {reason}")
        self.model = model
        self.reason = reason


class ModelNotFittedError(Exception):
    """Raised when predict/save/load is called before fit."""

    def __init__(self, model: str) -> None:
        super().__init__(f"{model} has not been fitted yet")
        self.model = model
