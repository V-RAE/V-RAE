from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if not name or name in self._factories:
            raise KeyError(f"Duplicate or empty {self.kind} registry key: {name!r}")
        self._factories[name] = factory

    def decorator(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def apply(factory: Callable[..., T]) -> Callable[..., T]:
            self.register(name, factory)
            return factory

        return apply

    def build(self, config: Mapping[str, Any], **kwargs: Any) -> T:
        name = str(config.get("name", ""))
        if name not in self._factories:
            raise KeyError(f"Unknown {self.kind} {name!r}; available: {sorted(self._factories)}")
        return self._factories[name](config=config, **kwargs)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._factories)


ENCODERS: Registry[Any] = Registry("encoder")
POOLERS: Registry[Any] = Registry("pooler")
DECODERS: Registry[Any] = Registry("decoder")
MODELS: Registry[Any] = Registry("model")
DIT_MODELS: Registry[Any] = Registry("DiT model")


def register_builtin_models() -> None:
    # Imports are local to keep the V-JEPA runtime dependency lazy.
    modules = (
        "vrae.models.autoencoder",
        "vrae.models.decoder",
        "vrae.models.pooling",
        "vrae.models.dit.video_dit",
        "vrae.models.encoders.vjepa2_1",
    )
    for module in modules:
        importlib.import_module(module)
