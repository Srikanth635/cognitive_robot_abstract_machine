"""Small ordered component registry used by the default pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Generic, Iterable, TypeVar

TComponent = TypeVar("TComponent")


@dataclass
class ComponentRegistry(Generic[TComponent]):
    """Ordered collection of pipeline components."""

    components: list[TComponent] = field(default_factory=list)

    def register(self, component: TComponent) -> None:
        self.components.append(component)

    def extend(self, components: Iterable[TComponent]) -> None:
        self.components.extend(components)

    def __iter__(self):
        return iter(self.components)
