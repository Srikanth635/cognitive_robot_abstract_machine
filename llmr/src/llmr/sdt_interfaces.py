"""
sdt_interfaces.py — Single boundary between llmr and semantic_digital_twin.

All SDT imports inside llmr MUST live here (as lazy imports inside function
bodies).  Every other llmr module calls the helpers below or uses WorldAdapter
— never ``from semantic_digital_twin…`` at module level.

Why: concentrating the coupling here means that when SDT changes its API only
this one file needs updating; the rest of llmr is untouched.

Pattern mirrors krrood: krrood has zero SDT imports and receives world data as
generic parameters/domains.  llmr follows the same principle — the
construction-time boundary (world_setup.py) is the only other accepted place
for SDT imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Structural Protocols — no SDT import at runtime
# ---------------------------------------------------------------------------


@runtime_checkable
class BodyLike(Protocol):
    """Minimal interface for SDT Body objects used inside llmr."""

    name: Any        # PrefixedName — has .name (str)
    global_pose: Any  # Pose — has .to_position() → x/y/z
    collision: Any   # ShapeCollection — has .as_bounding_box_collection_in_frame(…)


@runtime_checkable
class WorldLike(Protocol):
    """Minimal interface for SDT World objects used inside llmr."""

    bodies: List[Any]
    semantic_annotations: Iterable[Any]
    root: Any

    def get_semantic_annotations_by_type(self, t: Type) -> List: ...
    def get_semantic_annotations_of_body(self, body: Any) -> List: ...
    def get_bodies_by_name(self, name: Any) -> List: ...


# ---------------------------------------------------------------------------
# Attribute-chain helpers
# These are the ONLY place in llmr that knows about SDT's internal chaining.
# If SDT renames PrefixedName.name, Pose.to_position(), etc., fix it here.
# ---------------------------------------------------------------------------


def body_display_name(body: Any) -> str:
    """Return the display string for an SDT Body (hides the PrefixedName chain)."""
    name_obj = getattr(body, "name", None)
    if name_obj is None:
        return ""
    if hasattr(name_obj, "name"):
        return str(name_obj.name)
    return str(name_obj)


def body_xyz(body: Any) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, z) for any object with a .global_pose (hides Pose internals)."""
    try:
        pt = body.global_pose.to_position()
        return float(pt.x), float(pt.y), float(pt.z)
    except Exception:
        return None


def body_bounding_box_dims(
    body: Any,
    reference_frame: Any = None,
) -> Optional[Tuple[float, float, float]]:
    """Return (w, d, h) bounding box dimensions (hides ShapeCollection internals).

    :param body: Any object with a .collision ShapeCollection.
    :param reference_frame: Frame to compute dimensions in.  Defaults to *body* itself.
    """
    try:
        ref = reference_frame if reference_frame is not None else body
        bb = body.collision.as_bounding_box_collection_in_frame(ref).bounding_box()
        d = bb.dimensions
        return float(d[0]), float(d[1]), float(d[2])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WorldAdapter — only for operations that require a lazy SDT type import
# ---------------------------------------------------------------------------


class WorldAdapter:
    """Wraps an SDT World for operations that need a concrete SDT type at runtime.

    The simple World properties (bodies, semantic_annotations, root, etc.) are
    already stable SDT public API — callers should use them directly on the
    world object.  This adapter exists only for the two operations that need a
    lazy import of a concrete SDT class (AbstractRobot).
    """

    def __init__(self, world: "WorldLike") -> None:
        self._world = world

    def get_robot(self) -> Optional[Any]:
        """Return the first AbstractRobot annotation in the world, or None."""
        try:
            from semantic_digital_twin.robots.abstract_robot import AbstractRobot  # lazy
            robots = self._world.get_semantic_annotations_by_type(AbstractRobot)
            return robots[0] if robots else None
        except Exception:
            return None

    def find_manipulator(self, arm: Any) -> Optional[Any]:
        """Return the Manipulator for *arm*, or None."""
        robot = self.get_robot()
        if robot is None:
            return None
        try:
            if hasattr(robot, "get_manipulator_for_arm"):
                return robot.get_manipulator_for_arm(arm)
            if hasattr(robot, "manipulators"):
                manipulators = robot.manipulators
                if isinstance(manipulators, dict):
                    return manipulators.get(arm)
                if isinstance(manipulators, list) and manipulators:
                    return (
                        manipulators[arm.value]
                        if arm.value < len(manipulators)
                        else manipulators[0]
                    )
        except Exception:
            pass
        return None
