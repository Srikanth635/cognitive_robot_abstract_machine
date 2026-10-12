"""
Where the world injector finds the kinds a world has and the individuals living in it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph
from krrood.utils import recursive_subclasses
from typing_extensions import Any, List, Tuple

FOUNDATION_PACKAGE = "krrood"
"""
The package of the symbol graph, whose own classes are its machinery rather than kinds
of any world.
"""


class WorldSource(ABC):
    """
    The kinds and individuals a world has.
    """

    @abstractmethod
    def kinds(self) -> List[type]:
        """
        :return: Every kind loaded, each once.
        """

    @abstractmethod
    def individuals(self) -> List[Any]:
        """
        :return: Every live individual.
        """

    @abstractmethod
    def individuals_of(self, kind: type) -> List[Any]:
        """
        :return: The live individuals of a kind and of the kinds specialising it.
        """


@dataclass(frozen=True)
class SymbolGraphSource(WorldSource):
    """
    The world as krrood's symbol graph holds it: every loaded symbol class declared
    outside the excluded packages is a kind, and every live symbol of such a kind an
    individual.
    """

    excluded_packages: Tuple[str, ...] = (FOUNDATION_PACKAGE,)
    """
    The packages whose symbol classes are not kinds of the world, such as the symbol
    graph's own machinery or a component's description-level knowledge, which are
    symbols too but not things in the world.
    """

    def excludes(self, kind: type) -> bool:
        """
        :return: Whether a class is declared in one of the excluded packages.
        """
        return any(
            kind.__module__ == package or kind.__module__.startswith(f"{package}.")
            for package in self.excluded_packages
        )

    def kinds(self) -> List[type]:
        return [
            kind
            for kind in dict.fromkeys(recursive_subclasses(Symbol))
            if not self.excludes(kind)
        ]

    def individuals(self) -> List[Any]:
        return [
            wrapped.instance
            for wrapped in SymbolGraph().wrapped_instances
            if wrapped.instance is not None
            and not self.excludes(type(wrapped.instance))
        ]

    def individuals_of(self, kind: type) -> List[Any]:
        return [
            individual
            for individual in SymbolGraph().get_instances_of_type(kind)
            if individual is not None and not self.excludes(type(individual))
        ]
