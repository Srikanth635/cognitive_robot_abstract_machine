"""
Where the world injector finds the kinds a world has, the individuals living in it, and
the functions its world model declares.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph
from krrood.utils import recursive_subclasses
from typing_extensions import Any, Dict, List, Tuple

from llm_reasoner.world.functions import WorldFunction

FOUNDATION_PACKAGE = "krrood"
"""
The package of the symbol graph, whose own classes are its machinery rather than kinds
of any world.
"""


class WorldSource(ABC):
    """
    The kinds and individuals a world has, and the functions its world model declares.
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

    @abstractmethod
    def functions(self) -> List[WorldFunction]:
        """
        :return: Every function the world model declares, each once.
        """


@dataclass(frozen=True)
class SymbolGraphSource(WorldSource):
    """
    The world as krrood's symbol graph holds it: every loaded symbol class declared
    outside the excluded packages is a kind, every live symbol of such a kind an
    individual, and every function declared outside them with ``symbolic_function`` a
    function of the world model.
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
        return self.excludes_module(kind.__module__)

    def excludes_module(self, module_name: str) -> bool:
        """
        :return: Whether a module is one of the excluded packages or inside one.
        """
        return any(
            module_name == package or module_name.startswith(f"{package}.")
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

    def functions(self) -> List[WorldFunction]:
        declared: Dict[str, WorldFunction] = {}
        for module in list(sys.modules.values()):
            if not isinstance(module, ModuleType):
                continue
            for member in list(vars(module).values()):
                if not WorldFunction.is_declared(member):
                    continue
                if self.excludes_module(member.__module__):
                    continue
                function = WorldFunction.of(member)
                declared[function.identifier] = function
        return list(declared.values())
