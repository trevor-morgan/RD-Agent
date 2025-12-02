"""Tests for the model registry."""

from __future__ import annotations

import pytest

from rdagent_lab.core.registry import ModelRegistry, Registry


# --- Registry tests ---


def test_registry_register_decorator() -> None:
    """Should register a class with decorator."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("test_model")
    class TestModel:
        pass

    assert "test_model" in registry
    assert registry.get("test_model") is TestModel


def test_registry_register_without_name_uses_class_name() -> None:
    """Should use class name when no name provided."""
    registry: Registry = Registry("TestRegistry")

    @registry.register()
    class MyModel:
        pass

    assert "MyModel" in registry
    assert registry.get("MyModel") is MyModel


def test_registry_register_duplicate_raises() -> None:
    """Should raise on duplicate registration."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("dupe")
    class First:
        pass

    with pytest.raises(ValueError, match="already contains"):

        @registry.register("dupe")
        class Second:
            pass


def test_registry_create_instance() -> None:
    """Should create instance with kwargs."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("configurable")
    class Configurable:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    instance = registry.create("configurable", value=42)
    assert instance.value == 42


def test_registry_create_instance_default_kwargs() -> None:
    """Should create instance with default kwargs."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("with_defaults")
    class WithDefaults:
        def __init__(self, a: int = 1, b: str = "test") -> None:
            self.a = a
            self.b = b

    instance = registry.create("with_defaults")
    assert instance.a == 1
    assert instance.b == "test"


def test_registry_get_missing_raises() -> None:
    """Should raise KeyError for missing key."""
    registry: Registry = Registry("TestRegistry")
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent")


def test_registry_get_missing_shows_available() -> None:
    """Should show available options in error message."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("alpha")
    class Alpha:
        pass

    @registry.register("beta")
    class Beta:
        pass

    with pytest.raises(KeyError, match="alpha") as exc_info:
        registry.get("gamma")
    assert "beta" in str(exc_info.value)


def test_registry_list_returns_sorted() -> None:
    """Should return sorted list of registered names."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("zebra")
    class Z:
        pass

    @registry.register("alpha")
    class A:
        pass

    @registry.register("middle")
    class M:
        pass

    assert registry.list() == ["alpha", "middle", "zebra"]


def test_registry_contains() -> None:
    """Should support 'in' operator."""
    registry: Registry = Registry("TestRegistry")

    @registry.register("exists")
    class Exists:
        pass

    assert "exists" in registry
    assert "missing" not in registry


def test_registry_len() -> None:
    """Should return correct length."""
    registry: Registry = Registry("TestRegistry")
    assert len(registry) == 0

    @registry.register("one")
    class One:
        pass

    assert len(registry) == 1

    @registry.register("two")
    class Two:
        pass

    assert len(registry) == 2


# --- ModelRegistry tests ---


def test_model_registry_is_registry_instance() -> None:
    """Should be a Registry instance."""
    assert isinstance(ModelRegistry, Registry)
    assert ModelRegistry.name == "ModelRegistry"
