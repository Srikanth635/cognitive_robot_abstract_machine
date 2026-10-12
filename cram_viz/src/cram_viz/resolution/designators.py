"""
The predefined, underspecified designators the viewer offers: krrood match queries on
coraplex's pouring action, each written in a file of its own and evaluated on the
scene's world, stating some slots and leaving the others for the resolver.

A query reads krrood's factories, the classes in :data:`QUERY_VOCABULARY`, and the
scene's objects, each by its kind in snake case (``pouring_cup``, ``bowl``).
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from krrood.code_generation.naming import to_snake_case
from krrood.entity_query_language.factories import a, an, and_, not_, or_, the, variable
from krrood.entity_query_language.query.match import Match
from llm_reasoner.resolver.reflection import Slot, SlotKind, reflect_slots
from semantic_digital_twin.semantic_annotations.semantic_annotations import Tool
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Dict, List

from cram_viz.query_source import QuerySource

DESIGNATOR_DIRECTORY = Path(__file__).parent / "designator_queries"
"""
The directory holding the predefined designators' queries.
"""

QUERY_SUFFIX = ".py"
"""
The suffix of a designator's query file.
"""

QUERY_FACTORIES = (a, an, the, variable, and_, or_, not_)
"""
The krrood factories a designator's query reads, each by its name.
"""

QUERY_VOCABULARY = (PouringAction, Arms, Tool, Body)
"""
The classes a designator's query reads, each by its name.
"""

BUILTIN_NAMES = frozenset(dir(builtins))
"""
The names every query reads without a scene binding them.
"""

# %% the designators


class PouringDesignator(StrEnum):
    """
    The predefined pouring designators, each named for its query's file, in the order
    the viewer offers them.
    """

    OBJECTS_GIVEN = "objects_given"
    """
    The pouring cup and the bowl given; the arm and the tilt left open.
    """

    OBJECTS_OPEN = "objects_open"
    """
    Every slot left open, the source and the target included.
    """


@dataclass
class UnknownDesignator(ValueError):
    """
    No predefined designator has the name asked for.
    """

    identifier: str
    """
    The name asked for.
    """

    def __post_init__(self):
        super().__init__(f"no predefined designator is named '{self.identifier}'")


@dataclass
class NotAMatchQuery(TypeError):
    """
    A designator's query evaluates to something other than a krrood match.
    """

    identifier: str
    """
    The designator's name.
    """

    value: Any
    """
    What its query evaluated to.
    """

    def __post_init__(self):
        super().__init__(
            f"'{self.identifier}' evaluates to {type(self.value).__name__}, not a match"
        )


@dataclass
class DesignatorUnavailable(ValueError):
    """
    The scene cannot resolve a designator.
    """

    offered: OfferedDesignator
    """
    The designator as the scene offers it, stating what the scene lacks.
    """

    def __post_init__(self):
        lacks = [
            *(f"an object named {name}" for name in self.offered.unbound),
            *(f"a candidate for {slot}" for slot in self.offered.without_candidates),
        ]
        super().__init__(
            f"'{self.offered.identifier}' needs {', '.join(lacks)}, which the scene lacks"
        )


@dataclass(frozen=True)
class OfferedDesignator:
    """
    A predefined designator as the viewer offers it in one scene.
    """

    identifier: PouringDesignator
    """
    The designator's name.
    """

    query: str
    """
    The designator's match query, as written.
    """

    unbound: List[str]
    """
    The names the query reads that the scene binds to no object.
    """

    without_candidates: List[str]
    """
    The slots the query leaves to choose from candidates of which the scene has none.
    """

    @property
    def available(self) -> bool:
        """
        :return: Whether the scene can resolve the designator.
        """
        return not self.unbound and not self.without_candidates

    def payload(self) -> Dict[str, Any]:
        """
        :return: The offer as JSON data.
        """
        return dict(
            identifier=self.identifier,
            query=self.query,
            available=self.available,
            unbound=self.unbound,
            without_candidates=self.without_candidates,
        )


@dataclass(frozen=True)
class QuerySlots:
    """
    The slots of a designator's query, in the order the query states them.
    """

    names: List[str]
    """
    Every slot, by field name.
    """

    given: List[str]
    """
    The slots the query states a value for.
    """

    without_candidates: List[str]
    """
    The slots left to choose from candidates of which there are none.
    """

    @classmethod
    def of(cls, query: Match) -> QuerySlots:
        """
        :return: The slots of a match query.
        """
        slots = reflect_slots(query)
        return cls(
            names=[slot.attribute_name for slot in slots],
            given=[
                slot.attribute_name for slot in slots if slot.kind is SlotKind.BOUND
            ],
            without_candidates=[
                slot.attribute_name for slot in slots if cls.has_no_candidate(slot)
            ],
        )

    @staticmethod
    def has_no_candidate(slot: Slot) -> bool:
        """
        :return: Whether a slot is left to choose from candidates of which there are
            none.
        """
        return slot.kind is SlotKind.REFERENT and slot.domain == []


@dataclass(frozen=True)
class SceneNamespace:
    """
    The names a designator's query is evaluated with on a scene.
    """

    world: World
    """
    The scene's world.
    """

    @staticmethod
    def name_of(kind: type) -> str:
        """
        :return: The name a scene object of the given kind is read by.
        """
        return to_snake_case(kind.__name__)

    def names(self) -> Dict[str, Any]:
        """
        :return: krrood's factories and the vocabulary, each by its name, and the
            scene's objects, each by its kind; of several objects of one kind, the
            first.
        """
        objects: Dict[str, Any] = {}
        for annotation in self.world.semantic_annotations:
            objects.setdefault(self.name_of(type(annotation)), annotation)
        return {
            **objects,
            **{item.__name__: item for item in (*QUERY_FACTORIES, *QUERY_VOCABULARY)},
        }


@dataclass(frozen=True)
class DesignatorQuery:
    """
    A predefined designator: an underspecified krrood match query written in a file of
    its own.
    """

    identifier: PouringDesignator
    """
    The designator's name.
    """

    source: QuerySource
    """
    The designator's query.
    """

    @classmethod
    def shipped(cls, identifier: PouringDesignator) -> DesignatorQuery:
        """
        :return: The predefined designator of the given name, read from its file.
        """
        return cls(
            identifier=identifier,
            source=QuerySource.read(
                DESIGNATOR_DIRECTORY / f"{identifier}{QUERY_SUFFIX}"
            ),
        )

    @classmethod
    def named(cls, name: str) -> DesignatorQuery:
        """
        :return: The predefined designator of the given name.
        :raises UnknownDesignator: When none has that name.
        """
        if name not in PouringDesignator:
            raise UnknownDesignator(identifier=name)
        return cls.shipped(PouringDesignator(name))

    def query_in(self, names: Dict[str, Any]) -> Match:
        """
        :param names: The names the scene binds.
        :return: The designator's query on the scene.
        :raises NotAMatchQuery: When the query evaluates to something else.
        """
        query = self.source.value_in(dict(names))
        if not isinstance(query, Match):
            raise NotAMatchQuery(identifier=self.identifier, value=query)
        return query

    def unbound_in(self, names: Dict[str, Any]) -> List[str]:
        """
        :return: The names the query reads that the scene does not bind.
        """
        return sorted(self.source.names_read() - names.keys() - BUILTIN_NAMES)

    def offered_in(self, names: Dict[str, Any]) -> OfferedDesignator:
        """
        :param names: The names the scene binds.
        :return: The designator as the viewer offers it in the scene.
        """
        unbound = self.unbound_in(names)
        return OfferedDesignator(
            identifier=self.identifier,
            query=self.source.text,
            unbound=unbound,
            without_candidates=(
                []
                if unbound
                else QuerySlots.of(self.query_in(names)).without_candidates
            ),
        )
