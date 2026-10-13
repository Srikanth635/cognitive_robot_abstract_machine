"""
A small world of krrood symbols for the world injector: a body with geometry, inertial
properties and a pose, annotations referring to it, a predicate and functions about
bodies, and an action with a slot for a body. It imports only krrood, as a world built on krrood would
reach llm_reasoner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from krrood.entity_query_language.predicate import (
    Predicate,
    RenderedFields,
    Symbol,
    symbolic_function,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Noun,
    predicate_clause,
)
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
class IsHeavierThan(Predicate):
    """Whether a body weighs more than a given mass."""

    body: WorldBody
    """The body whose mass is compared."""

    mass: float
    """The mass compared against, in kilograms."""

    def __call__(self) -> bool:
        return self.body.inertial.mass > self.mass

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields):
        return predicate_clause(cls, Noun(fields["body"]), Noun(fields["mass"]))


@dataclass(eq=False)
class ActionWithBodyAndValueSlots(Symbol):
    """Tilt one body to pour from it."""

    source: WorldBody
    """The body poured from."""

    angle: float = 0.5
    """The angle to tilt the body to, in radians."""


class Side(Enum):
    """Which side of a body an action works on."""

    LEFT = 1
    RIGHT = 2


@dataclass(eq=False)
class ActionWithADefaultChoice(Symbol):
    """Work on one side of a body, a side being conventional."""

    source: WorldBody
    """The body worked on."""

    side: Side = Side.RIGHT
    """The side of the body worked on."""

    depth: Optional[float] = None
    """How deep to work into the body, in meters; nothing for the action's own choice."""

    support: Optional[WorldBody] = None
    """The body to rest the worked body on; nothing when it needs no support."""


@dataclass(eq=False)
class ActionPlannedByItsParent(Symbol):
    """An action whose plan its specialisations inherit."""

    source: WorldBody
    """The body acted on."""

    def plan(self) -> str:
        """
        :return: What the action does with its body.
        """
        return f"grasp {self.source.name}"


@dataclass(eq=False)
class ActionInheritingItsPlan(ActionPlannedByItsParent):
    """An action declaring nothing but what it inherits."""


# %% functions the world model declares

MARGIN_MEANING = "How much less it must weigh, in kilograms."
"""
What the docstring of :func:`weighs_less_than` says its margin is.
"""

UNSUPPORTED_REASON = "Needs a simulation"
"""
Why :func:`is_stable` cannot say.
"""


@dataclass(eq=False)
class DetachedPart(Symbol):
    """A part of the world that exists only where a test attaches one."""

    attached_to: Optional[WorldBody] = None
    """The body this part is attached to."""


@symbolic_function
def weighs_less_than(body: WorldBody, other: WorldBody, margin: float = 0.0) -> bool:
    """
    Whether a body weighs less than another.

    :param body: The body that may weigh less.
    :param other: The body it is compared with.
    :param margin: How much less it must weigh, in kilograms.
    :return: Whether the body weighs less than the other by the margin.
    """
    return body.inertial.mass + margin < other.inertial.mass


@symbolic_function
def is_attached(part: DetachedPart) -> bool:
    """
    Whether a part is attached to a body.

    :param part: The part asked about.
    :return: Whether the part is attached.
    """
    return part.attached_to is not None


@symbolic_function
def is_stable(body: WorldBody) -> bool:
    """
    Whether a body stays where it is.

    :param body: The body asked about.
    :return: Whether the body stays.
    """
    raise NotImplementedError(UNSUPPORTED_REASON)


@symbolic_function
def fits_inside(body: WorldBody, box: BoxShape) -> bool:
    """
    Whether a body fits inside a box, which is no individual of the world.

    :param body: The body asked about.
    :param box: The box it must fit inside.
    :return: Whether the body fits.
    """
    return any(shape.width <= box.width for shape in body.shapes)


def weighs_more_than(body: WorldBody, other: WorldBody) -> bool:
    """
    Whether a body weighs more than another, a function nobody declared symbolic.

    :param body: The body that may weigh more.
    :param other: The body it is compared with.
    :return: Whether the body weighs more.
    """
    return body.inertial.mass > other.inertial.mass


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


def body_moved_by_a_readable_joint(joint_position: List[float]) -> WorldBody:
    """
    :param joint_position: Holds the current joint position; changing its only element
        moves the joint.
    :return: A body whose position depends on a joint the world can read.
    """
    joint = FloatVariable.create_with_resolver("lid_joint", lambda: joint_position[0])
    return WorldBody(name="lid", position=Vector([joint, 0.0, 0.0]))
