"""
The world injector as other components reach it: the version of this contract, the names
it shares with them, and what a component may ask of a world.

A component such as the resolver imports from this module only, so that it depends on
the contract rather than on how a world answers it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from typing_extensions import Any, List, Optional

from llm_reasoner.clients.tools import Tool
from llm_reasoner.world.classes import identifier_of, is_individual
from llm_reasoner.world.describer import IndividualDescriber
from llm_reasoner.world.holders import Holder
from llm_reasoner.world.naming import HANDLE_SEPARATOR
from llm_reasoner.world.rendering import IndividualRenderer

__all__ = [
    "HANDLE_SEPARATOR",
    "WORLD_ACCESS_VERSION",
    "Holder",
    "WorldAccess",
    "WorldToolName",
    "identifier_of",
    "is_individual",
    "text_description",
]

WORLD_ACCESS_VERSION = "1.0"
"""
The version of this contract: of the operations below and of the definitions and answer
shapes of the world tools.
"""


class WorldToolName(StrEnum):
    """
    The world injector's tools, each named for the question it answers.
    """

    LIST_KINDS = "list_kinds"
    """
    Which kinds the world has.
    """

    DESCRIBE_KIND = "describe_kind"
    """
    What a kind is and what its individuals hold.
    """

    FIND_INDIVIDUALS = "find_individuals"
    """
    Which individuals of a kind meet conditions.
    """

    DESCRIBE_INDIVIDUAL = "describe_individual"
    """
    What an individual is and holds.
    """

    HOLDERS_OF = "holders_of"
    """
    Which individuals hold an individual.
    """

    FOLLOW_PATH = "follow_path"
    """
    What a path of fields reaches from an individual.
    """

    LIST_PREDICATES = "list_predicates"
    """
    Which predicates and value functions the world model declares.
    """

    EVALUATE_PREDICATE = "evaluate_predicate"
    """
    What a predicate or value function says about given arguments.
    """


class WorldAccess(ABC):
    """
    What a component may ask of a world: the handles naming its individuals, what holds
    an individual, and the tools a language model asks it with.
    """

    @property
    def version(self) -> str:
        """
        The version of the contract this world fulfils.
        """
        return WORLD_ACCESS_VERSION

    @abstractmethod
    def handle_of(self, individual: Any) -> str:
        """
        :return: The handle naming an individual, the same for the world's lifetime.
        """

    @abstractmethod
    def individual_of(self, handle: str) -> Any:
        """
        :return: The individual a handle names.
        :raises UnknownHandle: When the handle was never given out.
        :raises ExpiredHandle: When its individual no longer exists.
        """

    @abstractmethod
    def is_live(self, handle: str) -> bool:
        """
        :return: Whether a handle was given out to an individual that still exists.
        """

    @abstractmethod
    def holders(self, handle: str) -> List[Holder]:
        """
        :return: The individuals holding the individual a handle names.
        """

    @abstractmethod
    def tools(self) -> List[Tool]:
        """
        :return: The tools a language model asks this world with, named by
            :class:`WorldToolName`.
        """


def text_description(individual: Any, line_budget: Optional[int] = None) -> str:
    """
    :param individual: An individual of the world.
    :param line_budget: The most lines the description takes, the renderer's own limit
        if ``None``.
    :return: The individual described as text for reading, the individuals it holds
        numbered within the description rather than by handle.
    """
    renderer = (
        IndividualRenderer()
        if line_budget is None
        else IndividualRenderer(line_budget=line_budget)
    )
    return renderer.render(IndividualDescriber().describe(individual))
