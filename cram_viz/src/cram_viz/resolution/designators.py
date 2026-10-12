"""
The predefined, underspecified pouring designators the viewer offers: each a query on
coraplex's pouring action written on the scene's world, giving some slots and leaving
the others for the resolver, some with a task stating what the pouring is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from krrood.entity_query_language.factories import an, and_, the, variable
from krrood.entity_query_language.query.match import Match
from llm_reasoner.resolver.reflection import reflect_slots
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    PouringCup,
    Tool,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Callable, Dict, List, Optional, Tuple

SEVERAL = 3
"""
How many alternatives a variant asking for several ways to pour asks for.
"""

GEOMETRY_SLOTS = (
    "pour_side",
    "pour_side_offset",
    "pour_approach_offset",
    "pour_height",
)
"""
The pouring action's fields stating where the pour happens, left open by the variant
asking for the pouring geometry.
"""

GENTLE_TILT = (1.0, 1.5)
"""
The tilt angles, in radians, a variant asking for a gentle tilt allows.
"""

# %% queries on the scene


@dataclass(frozen=True)
class SceneQueries:
    """
    Writes pouring queries on a scene's world.
    """

    world: World
    """
    The scene's world.
    """

    def has(self, kind: type[HasRootBody]) -> bool:
        """
        :return: Whether the scene holds an object annotated as the given kind.
        """
        return any(
            isinstance(annotation, kind)
            for annotation in self.world.semantic_annotations
        )

    def annotation(self, kind: type[HasRootBody]) -> HasRootBody:
        """
        :return: The scene's first object annotated as the given kind.
        """
        return next(
            annotation
            for annotation in self.world.semantic_annotations
            if isinstance(annotation, kind)
        )

    def vessel(self) -> Any:
        """
        :return: A variable over the scene's pouring vessel.
        """
        return variable(Tool, domain=[self.annotation(PouringCup)])

    def bowl(self) -> Any:
        """
        :return: A variable over the body of the scene's bowl.
        """
        return variable(Body, domain=[self.annotation(Bowl).root])

    @staticmethod
    def pouring_slots() -> Dict[str, Any]:
        """
        :return: A pouring action's usual open slots: any body as the target, any tool
            as the source, the arm and the tilt angle.
        """
        return dict(
            target_container=variable(Body),
            source_container=variable(Tool),
            arm=...,
            tilt_angle=...,
        )


# %% the variants


class PouringVariant(StrEnum):
    """
    The predefined pouring designators, each named for what it gives.
    """

    INTO_THE_BOWL = "into_the_bowl"
    FROM_THE_VESSEL = "from_the_vessel"
    VESSEL_INTO_BOWL = "vessel_into_bowl"
    ANYTHING = "anything"
    LEFT_ARM_GENTLE_TILT = "left_arm_gentle_tilt"
    SEVERAL_WAYS = "several_ways"
    GEOMETRY = "geometry"
    UNTIL_HALF_FULL = "until_half_full"
    ALL_OF_IT = "all_of_it"
    UNTIL_TOLD_TO_STOP = "until_told_to_stop"


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
class DesignatorUnavailable(ValueError):
    """
    The scene lacks objects a predefined designator needs.
    """

    identifier: str
    """
    The designator's name.
    """

    missing: List[str]
    """
    The kinds of object the scene lacks.
    """

    def __post_init__(self):
        super().__init__(
            f"'{self.identifier}' needs {', '.join(self.missing)}, which the scene lacks"
        )


@dataclass(frozen=True)
class OfferedDesignator:
    """
    A predefined designator as the viewer offers it in one scene.
    """

    identifier: PouringVariant
    """
    The designator's name.
    """

    label: str
    """
    What the designator is, as its chip reads.
    """

    task: str
    """
    What the pouring is for, none when empty.
    """

    available: bool
    """
    Whether the scene holds every object the designator needs.
    """

    missing: List[str]
    """
    The kinds of object the designator needs that the scene lacks.
    """

    given: List[str]
    """
    The slots the designator gives, empty when it is unavailable.
    """

    left_open: List[str]
    """
    The slots the designator leaves open, empty when it is unavailable.
    """


@dataclass(frozen=True)
class SlotSplit:
    """
    Which slots of a query are given and which are left open.
    """

    given: List[str]
    """
    The slots given, by field name: a value, or one individual to take.
    """

    left_open: List[str]
    """
    The slots left for the resolver, by field name.
    """


def gentle_tilt(action: Any) -> List[Any]:
    """
    :return: The conditions of a gentle tilt on a pouring action's variable.
    """
    return [
        and_(
            action.tilt_angle >= GENTLE_TILT[0],
            action.tilt_angle <= GENTLE_TILT[1],
        )
    ]


@dataclass(frozen=True)
class DesignatorVariant:
    """
    A predefined, underspecified pouring designator: the slots it gives, the slots it
    leaves open beyond the usual, the conditions it sets, and what the pouring is for.
    """

    identifier: PouringVariant
    """
    The variant's name.
    """

    label: str
    """
    What the variant is, as its chip reads.
    """

    gives: Callable[[SceneQueries], Dict[str, Any]] = field(default=lambda q: {})
    """
    The slots the variant gives on a scene, by field name.
    """

    also_open: Tuple[str, ...] = ()
    """
    Fields left open beyond a pouring action's usual open slots.
    """

    conditions: Optional[Callable[[Any], List[Any]]] = None
    """
    Writes the conditions the variant sets on the action's variable, none when absent.
    """

    alternatives: int = 1
    """
    How many alternative actions the variant asks for.
    """

    requires: Tuple[type[HasRootBody], ...] = field(default=(PouringCup,))
    """
    The kinds of object the scene must hold for the variant to be resolved.
    """

    task: str = ""
    """
    What the pouring is for, none when empty.
    """

    def query(self, queries: SceneQueries) -> Match:
        """
        :return: The variant's query on a scene: the usual open slots, the ones it also
            leaves open, and the ones it gives.
        """
        slots = {
            **queries.pouring_slots(),
            **{name: ... for name in self.also_open},
            **self.gives(queries),
        }
        if self.alternatives > 1:
            query = an(PouringAction)(**slots)
            query.expression.limit(self.alternatives)
        else:
            query = the(PouringAction)(**slots)
        if self.conditions is None:
            return query
        return query.where(*self.conditions(query.variable))

    def missing_in(self, queries: SceneQueries) -> List[type[HasRootBody]]:
        """
        :return: The kinds of object the variant needs that the scene lacks.
        """
        return [kind for kind in self.requires if not queries.has(kind)]

    def offered_in(self, queries: SceneQueries) -> OfferedDesignator:
        """
        :return: The designator as the viewer offers it in the scene.
        """
        missing = [kind.__name__ for kind in self.missing_in(queries)]
        split = SlotSplit(given=[], left_open=[]) if missing else self.slots(queries)
        return OfferedDesignator(
            identifier=self.identifier,
            label=self.label,
            task=self.task,
            available=not missing,
            missing=missing,
            given=split.given,
            left_open=split.left_open,
        )

    def slots(self, queries: SceneQueries) -> SlotSplit:
        """
        :return: Which slots of the variant's query it gives and which it leaves open.
        """
        given = list(self.gives(queries))
        return SlotSplit(
            given=given,
            left_open=[
                slot.attribute_name
                for slot in reflect_slots(self.query(queries))
                if slot.attribute_name not in given
            ],
        )


def into_the_bowl(queries: SceneQueries) -> Dict[str, Any]:
    """
    :return: The bowl as the target.
    """
    return dict(target_container=queries.bowl())


def from_the_vessel(queries: SceneQueries) -> Dict[str, Any]:
    """
    :return: The pouring vessel as the source.
    """
    return dict(source_container=queries.vessel())


def from_the_vessel_into_the_bowl(queries: SceneQueries) -> Dict[str, Any]:
    """
    :return: The pouring vessel as the source and the bowl as the target.
    """
    return {**from_the_vessel(queries), **into_the_bowl(queries)}


POURING_VARIANTS = (
    DesignatorVariant(
        PouringVariant.INTO_THE_BOWL,
        "pour into the bowl",
        gives=into_the_bowl,
        requires=(PouringCup, Bowl),
    ),
    DesignatorVariant(
        PouringVariant.FROM_THE_VESSEL,
        "pour from the pouring vessel",
        gives=from_the_vessel,
    ),
    DesignatorVariant(
        PouringVariant.VESSEL_INTO_BOWL,
        "pour from the vessel into the bowl",
        gives=from_the_vessel_into_the_bowl,
        requires=(PouringCup, Bowl),
    ),
    DesignatorVariant(PouringVariant.ANYTHING, "pour, everything open"),
    DesignatorVariant(
        PouringVariant.LEFT_ARM_GENTLE_TILT,
        "pour with the left arm, tilting gently",
        gives=lambda q: dict(arm=Arms.LEFT),
        conditions=gentle_tilt,
    ),
    DesignatorVariant(
        PouringVariant.SEVERAL_WAYS, "three ways to pour", alternatives=SEVERAL
    ),
    DesignatorVariant(
        PouringVariant.GEOMETRY,
        "pouring geometry open",
        gives=from_the_vessel_into_the_bowl,
        also_open=GEOMETRY_SLOTS,
        requires=(PouringCup, Bowl),
    ),
    DesignatorVariant(
        PouringVariant.UNTIL_HALF_FULL,
        "into the bowl until it is half full",
        gives=from_the_vessel_into_the_bowl,
        requires=(PouringCup, Bowl),
        task="Pour the milk into the bowl until the bowl is half full.",
    ),
    DesignatorVariant(
        PouringVariant.ALL_OF_IT,
        "into the bowl, all of it",
        gives=from_the_vessel_into_the_bowl,
        requires=(PouringCup, Bowl),
        task="Pour all of the milk into the bowl.",
    ),
    DesignatorVariant(
        PouringVariant.UNTIL_TOLD_TO_STOP,
        "into the bowl until told to stop",
        gives=from_the_vessel_into_the_bowl,
        requires=(PouringCup, Bowl),
        task="Pour the milk into the bowl until I tell you to stop.",
    ),
)
"""
Every predefined pouring designator, in the order the viewer offers them.
"""


def variant_named(identifier: str) -> DesignatorVariant:
    """
    :return: The predefined designator of the given name.
    :raises UnknownDesignator: When none has that name.
    """
    named = [
        variant for variant in POURING_VARIANTS if variant.identifier == identifier
    ]
    if not named:
        raise UnknownDesignator(identifier=identifier)
    return named[0]
