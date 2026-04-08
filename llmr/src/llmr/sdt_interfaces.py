"""
sdt_interfaces.py — Single boundary between llmr and semantic_digital_twin.

All SDT imports inside llmr MUST live here.  Every other llmr module must
reference SDT types only through the Protocols defined in this file or through
the WorldAdapter helper, never via direct ``from semantic_digital_twin…`` imports
at module-level.

Why: concentrating the coupling here means that when SDT changes its API only
this one file needs to be updated; the rest of llmr remains untouched.

Pattern mirrors krrood: krrood has zero SDT imports and receives world data as
generic parameters/domains.  llmr follows the same principle for its core
logic; the construction-time boundary (world_setup.py) is the only other
accepted place for SDT imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Type

# ---------------------------------------------------------------------------
# Structural Protocols — no SDT import needed at runtime
# ---------------------------------------------------------------------------
# These describe the shapes that llmr actually needs from SDT objects.
# SDT's real classes satisfy them via duck typing; no isinstance() checks
# against SDT types are required outside this file.

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]


@runtime_checkable
class BodyLike(Protocol):
    """Minimal interface for SDT Body objects used inside llmr."""

    name: Any
    global_pose: Any   #
    collision: Any


@runtime_checkable
class WorldLike(Protocol):
    """Minimal interface for SDT World objects used inside llmr."""

    bodies: List[Any]
    semantic_annotations: Iterable[Any]
    root: Any

    def get_semantic_annotations_by_type(self, t: Type) -> List: ...
    def get_semantic_annotations_of_body(self, body: Any) -> List: ...


# ---------------------------------------------------------------------------
# WorldAdapter — wraps an SDT World
# ---------------------------------------------------------------------------


class WorldAdapter:
    """Thin facade over an SDT World instance.

    All methods that need a concrete SDT type (e.g. AbstractRobot) perform
    **lazy imports** inside their bodies so that importing this module never
    triggers an SDT import at module-load time.

    Usage::

        adapter = WorldAdapter(world)
        robot   = adapter.get_robot()
        manip   = adapter.find_manipulator(arm)
    """

    def __init__(self, world: "WorldLike") -> None:
        self._world = world

    # -- body / annotation queries ------------------------------------------

    @property
    def bodies(self) -> List[Any]:
        return self._world.bodies

    @property
    def semantic_annotations(self) -> Iterable[Any]:
        return self._world.semantic_annotations

    @property
    def root(self) -> Any:
        return self._world.root

    def get_by_type(self, annotation_type: Type) -> List:
        return self._world.get_semantic_annotations_by_type(annotation_type)

    def get_annotations_of_body(self, body: Any) -> List:
        return self._world.get_semantic_annotations_of_body(body)

    # -- robot helpers (lazy SDT imports) -----------------------------------

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
