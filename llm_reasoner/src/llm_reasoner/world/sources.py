"""
Where the world injector finds the kinds a world has and the individuals living in it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph
from krrood.utils import recursive_subclasses
from typing_extensions import Any, List


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
    The world as krrood's symbol graph holds it: every loaded symbol class is a kind,
    and every live symbol an individual.
    """

    def kinds(self) -> List[type]:
        return list(dict.fromkeys([Symbol, *recursive_subclasses(Symbol)]))

    def individuals(self) -> List[Any]:
        return [
            wrapped.instance
            for wrapped in SymbolGraph().wrapped_instances
            if wrapped.instance is not None
        ]

    def individuals_of(self, kind: type) -> List[Any]:
        return [
            individual
            for individual in SymbolGraph().get_instances_of_type(kind)
            if individual is not None
        ]
