"""
A small, world-free domain for exercising the backend.

It mirrors the shape of a real robot designator -- enums, a boolean, a continuous
parameter, a reference to an individual, and a nested description -- without depending
on a simulator, a robot, or a semantic digital twin. Everything the backend does with
``coraplex``'s ``GraspDescription`` it does with ``Grasp`` here.

``Gripper`` and ``Container`` subclass krrood's ``Symbol``, so every instance registers
itself in the ``SymbolGraph`` on construction. That is what lets ``variable(Gripper)``
find its extension with no domain argument and no import of this module by the backend.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol


class ApproachDirection(enum.Enum):
    """Which side the gripper comes in from."""

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"


class VerticalAlignment(enum.Enum):
    """How the gripper is aligned along the vertical axis."""

    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class Gripper(Symbol):
    """An end effector that can be commanded to open and close."""

    name: str
    """Human-readable name, e.g. 'left_gripper'."""

    max_opening: float = 0.09
    """Widest the jaws open, in metres."""

    def __repr__(self) -> str:
        return f"Gripper(name={self.name!r}, max_opening={self.max_opening})"


@dataclass
class Container(Symbol):
    """Something on a surface that a robot might pick up."""

    name: str
    """Human-readable name."""

    width: float = 0.07
    """Widest horizontal extent, in metres."""

    has_handle: bool = False
    """Whether it has a handle that affords a side grasp."""

    def __repr__(self) -> str:
        return (
            f"Container(name={self.name!r}, width={self.width}, "
            f"has_handle={self.has_handle})"
        )


@dataclass
class Grasp:
    """How to close a gripper around a body."""

    approach_direction: ApproachDirection
    """The direction the gripper approaches the object from."""

    vertical_alignment: VerticalAlignment
    """Whether the gripper aligns with the top or the bottom of the object."""

    gripper: Gripper
    """The end effector that performs the grasp."""

    rotate_gripper: bool = False
    """Rotate the gripper by 90 degrees before closing."""

    manipulation_offset: float = 0.05
    """Distance in metres between the object centre and the gripper's target pose."""


@dataclass
class PickUp:
    """Pick a container up off its supporting surface."""

    target: Container
    """The object to be picked up."""

    grasp: Grasp
    """How to grasp it."""

    lift_height: float = 0.1
    """How far to lift the object after the gripper closes, in metres."""


#: Keeps the scene's individuals alive.
#:
#: krrood's ``SymbolGraph`` holds its members weakly, so an individual that nothing
#: else references is collected and silently disappears from ``variable(T)``'s
#: extension. Holding a strong reference is the caller's job -- in a real system the
#: world holds it; here this list does.
_SCENE: list = []


def build_kitchen() -> dict:
    """
    Populate a tiny scene.

    :return: The named individuals, which are also discoverable through the
        ``SymbolGraph`` via ``variable(Gripper)`` and ``variable(Container)``.
    """
    scene = {
        "left_gripper": Gripper(name="left_gripper", max_opening=0.09),
        "right_gripper": Gripper(name="right_gripper", max_opening=0.05),
        "mug": Container(name="ceramic_mug", width=0.08, has_handle=True),
        "bowl": Container(name="cereal_bowl", width=0.16, has_handle=False),
        "glass": Container(name="drinking_glass", width=0.06, has_handle=False),
    }
    _SCENE.extend(scene.values())
    return scene