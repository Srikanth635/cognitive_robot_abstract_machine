"""
A small world of krrood symbols for describing individuals: a body with geometry, inertial
properties and a pose, annotations referring to it, and an action with a slot for a body.
It imports only krrood, as a world built on krrood would reach llm_reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.field_metadata import JSONMetadata
from krrood.symbolic_math.symbolic_math import FloatVariable, Vector
from typing_extensions import List, Optional


@dataclass(eq=False)
class BoxShape:
    """A box-shaped piece of geometry."""

    width: float
    """The extent along the x axis, in metres."""

    height: float
    """The extent along the z axis, in metres."""


@dataclass(eq=False)
class InertialProperties:
    """The inertial properties of a body."""

    mass: float = 1.0
    """The mass of the body in kilograms."""

    centre_of_mass: Vector = field(default_factory=lambda: Vector([0.0, 0.0, 0.05]))
    """The centre of mass in the frame of the body."""


@dataclass(eq=False)
class WorldBody(Symbol):
    """A rigid piece of the world."""

    name: str
    """The name identifying the body."""

    shapes: List[BoxShape] = field(default_factory=list)
    """The shapes the body collides with."""

    inertial: InertialProperties = field(default_factory=InertialProperties)
    """The inertial properties of the body."""

    position: Optional[Vector] = None
    """Where the body is in the world."""

    attached_to: Optional[WorldBody] = None
    """The body this one is attached to."""

    identifier: UUID = field(default_factory=uuid4)
    """The identifier of the body."""

    tags: List[str] = field(default_factory=list)
    """Free-form tags."""

    cached_geometry: List[BoxShape] = field(
        default_factory=list, metadata=JSONMetadata(serialize=False).as_dict()
    )
    """Geometry derived from the shapes, kept for speed."""

    _index: int = 0
    """The position of the body in its world's bookkeeping."""


@dataclass(eq=False)
class BodyAnnotation(Symbol):
    """A meaning given to a body."""

    body: WorldBody
    """The body the annotation gives a meaning to."""


@dataclass(eq=False)
class VesselAnnotation(BodyAnnotation):
    """A vessel people drink from."""


# Left without a docstring on purpose: it stands for a class that documents nothing of
# its own while its ancestor does.
@dataclass(eq=False)
class UndocumentedAnnotation(BodyAnnotation): ...


@dataclass(eq=False)
class SymbolFillingFieldsLater(Symbol):
    """A symbol that fills in some of its fields itself, after it is made."""

    filled_later: int = field(init=False)
    """A value the symbol sets once it is in use."""


@dataclass(eq=False)
class ActionWithBodySlot(Symbol):
    """Pour from one body."""

    source: WorldBody
    """The body poured from."""


def body_with_geometry() -> WorldBody:
    """
    :return: A body with one box, a set mass, a constant position and a default centre of
        mass.
    """
    return WorldBody(
        name="jug",
        shapes=[BoxShape(width=0.1, height=0.2)],
        inertial=InertialProperties(mass=0.4),
        position=Vector([1.0, 2.0, 0.8]),
    )


def body_moved_by_a_joint() -> WorldBody:
    """
    :return: A body whose position depends on a joint of the world.
    """
    return WorldBody(
        name="drawer", position=Vector([FloatVariable(name="drawer_joint"), 0.0, 0.0])
    )
