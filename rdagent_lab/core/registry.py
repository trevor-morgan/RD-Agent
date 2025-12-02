"""Simple registries for components."""

from typing import TypeVar, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    """Lightweight registry for component discovery and instantiation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str | None = None):
        """Decorator to register a class."""

        def decorator(cls: type[T]) -> type[T]:
            key = name or cls.__name__
            if key in self._registry:
                raise ValueError(f"{self.name} already contains '{key}'")
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Get a registered class by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(f"'{name}' not found in {self.name}. Available: {available}")
        return self._registry[name]

    def create(self, name: str, **kwargs) -> T:
        """Create an instance of a registered class."""
        cls = self.get(name)
        return cls(**kwargs)

    def list(self) -> list[str]:
        """List all registered names."""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


ModelRegistry = Registry("ModelRegistry")
StrategyRegistry = Registry("StrategyRegistry")
AgentRegistry = Registry("AgentRegistry")
