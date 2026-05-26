"""Scene graph gateway — single access point for every SymbolGraph and SDT query in agentic_llmr."""

from __future__ import annotations

from typing_extensions import Any, List, Optional, Set, Tuple

from krrood.symbol_graph.symbol_graph import SymbolGraph

# Robot annotation class names used for MRO-based type discrimination throughout
# the package. A single frozenset here avoids scattered string literals.
_SEMANTIC_ANNOTATION_TYPE_NAMES = frozenset(
    {
        "AbstractRobot",
        "SemanticRobotAnnotation",
        "KinematicChain",
        "Manipulator",
        "ParallelGripper",
        "HumanoidGripper",
        "Sensor",
        "Camera",
        "Base",
        "Torso",
        "Neck",
        "Arm",
        "Finger",
    }
)


# ── Active-world helper ────────────────────────────────────────────────────────


def _get_active_world_or_none() -> Optional[Any]:
    """Return the current active SDT world, or None if unavailable."""
    try:
        from agentic_llmr.integrations.world_manager import get_active_world
        world, _ = get_active_world()
        return world
    except Exception:
        return None


# ── Robot annotation predicate ─────────────────────────────────────────────────


def _is_robot_annotation(instance: Any) -> bool:
    """Return True if *instance* is a robot structural annotation (Arm, Gripper, etc.)."""
    try:
        mro_names = {cls.__name__ for cls in type(instance).__mro__}
    except AttributeError:
        return False
    if mro_names & _SEMANTIC_ANNOTATION_TYPE_NAMES:
        return True
    return hasattr(instance, "_robot") and (
        hasattr(instance, "root")
        or hasattr(instance, "tool_frame")
        or hasattr(instance, "tip")
    )


# ── Duck-typed body helpers ────────────────────────────────────────────────────


def symbol_display_name(body: Any) -> str:
    """Return a clean display name for a body instance (hides PrefixedName chain)."""
    name_obj = getattr(body, "name", None)
    if name_obj is None:
        return ""
    if hasattr(name_obj, "name"):
        return str(name_obj.name)
    return str(name_obj)


def symbol_bounding_box(
    body: Any,
    reference_frame: Optional[Any] = None,
) -> Optional[Tuple[float, float, float]]:
    """Return (depth, width, height) bounding box dims, or None if unavailable."""
    try:
        ref = reference_frame if reference_frame is not None else body
        dims = (
            body.collision.as_bounding_box_collection_in_frame(ref)
            .bounding_box()
            .dimensions
        )
        return float(dims[0]), float(dims[1]), float(dims[2])
    except Exception:
        return None


# ── SymbolGraph entity discovery ───────────────────────────────────────────────


def get_annotations(symbol_graph: Optional[SymbolGraph] = None) -> List[Any]:
    """Return semantic annotation instances from the SymbolGraph for the active world.

    Covers both environmental annotations (Milk, Table — have .bodies) and
    robot-structural annotations (Arm, ParallelGripper — match
    _SEMANTIC_ANNOTATION_TYPE_NAMES via MRO).

    Filters by active world so stale entries from earlier kernel runs are excluded.
    """
    try:
        graph = symbol_graph or SymbolGraph()
    except Exception:
        return []
    active_world = _get_active_world_or_none()
    result: List[Any] = []
    seen: Set[int] = set()
    for wrapped in graph.wrapped_instances:
        inst = wrapped.instance
        if inst is None or id(inst) in seen:
            continue
        seen.add(id(inst))
        if active_world is not None and getattr(inst, "_world", None) is not active_world:
            continue
        mro_names = {cls.__name__ for cls in type(inst).__mro__}
        if hasattr(inst, "bodies") or (mro_names & _SEMANTIC_ANNOTATION_TYPE_NAMES):
            result.append(inst)
    return result


def get_bodies(symbol_graph: Optional[SymbolGraph] = None) -> List[Any]:
    """Return physical body (link) instances from the SymbolGraph for the active world.

    Body subclasses KinematicStructureEntity → WorldEntity(Symbol), so Body IS a
    Symbol subclass. We iterate wrapped_instances and discriminate by MRO name "Body"
    for SDT-independence.

    Filters by active world so stale SymbolGraph entries from prior world loads
    in the same kernel session are excluded — prevents cross-world FK failures.
    """
    try:
        graph = symbol_graph or SymbolGraph()
    except Exception:
        return []
    active_world = _get_active_world_or_none()
    result: List[Any] = []
    seen: Set[int] = set()
    for wrapped in graph.wrapped_instances:
        inst = wrapped.instance
        if inst is None or id(inst) in seen:
            continue
        seen.add(id(inst))
        if active_world is not None and getattr(inst, "_world", None) is not active_world:
            continue
        if "Body" in {cls.__name__ for cls in type(inst).__mro__}:
            result.append(inst)
    return result


# ── Kinematics bridge ──────────────────────────────────────────────────────────


def compute_grasp_descriptions(
    manipulator: Any,
    position_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    world_root: Any,
) -> Any:
    """Bridge function: construct SDT Pose internally and delegate to GraspDescription.

    Callers pass plain Python tuples — no SDT spatial type imports needed outside bridge.
    """
    from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion
    from pycram.datastructures.grasp import GraspDescription
    x, y, z = position_xyz
    qx, qy, qz, qw = quat_xyzw
    pose = Pose(
        position=Point3(x, y, z),
        orientation=Quaternion(qx, qy, qz, qw),
        reference_frame=world_root,
    )
    return GraspDescription.calculate_grasp_descriptions(manipulator, pose)


def sort_by_volume(annotations: List[Any], largest_first: bool = True) -> List[Any]:
    """Sort semantic annotations by bounding-box volume. Falls back to original order on error."""
    def _volume(ann: Any) -> float:
        root = getattr(ann, "root", None)
        if root is None:
            return 0.0
        dims = symbol_bounding_box(root)
        return float(dims[0]) * float(dims[1]) * float(dims[2]) if dims else 0.0

    try:
        return sorted(annotations, key=_volume, reverse=largest_first)
    except Exception:
        return list(annotations)


# ── Robot helpers ──────────────────────────────────────────────────────────────


def get_arm_label(arm: Any, robot_view: Any) -> str:
    """Return 'left', 'right', or the arm's name — resolved from the robot's own geometry.

    Uses the robot view's left_arm / right_arm references first, then falls back to
    name-string heuristics so any robot model is handled without hardcoding.
    """
    left_arm = getattr(robot_view, "left_arm", None)
    right_arm = getattr(robot_view, "right_arm", None)
    if left_arm is not None and arm is left_arm:
        return "left"
    if right_arm is not None and arm is right_arm:
        return "right"
    name_obj = getattr(arm, "name", None)
    arm_name = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj) if name_obj else ""
    lower = arm_name.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return arm_name or "arm"


def find_body_by_name(name: str) -> Optional[Any]:
    """Return the first Body in the active world whose display name matches *name*, or None."""
    for body in get_bodies():
        if symbol_display_name(body) == name:
            return body
    return None


def get_robot_base_body(robot_view: Any) -> Optional[Any]:
    """Return the robot's base/root Body from the robot view annotation.

    Walks 'base', 'root', 'chassis' attributes on the robot view. If the candidate
    is itself a robot sub-annotation (Base, Torso), dereferences to its .root Body.
    Returns None when no suitable Body is found.
    """
    if robot_view is None:
        return None
    for attr in ("base", "root", "chassis"):
        candidate = getattr(robot_view, attr, None)
        if candidate is None:
            continue
        if not _is_robot_annotation(candidate) and hasattr(candidate, "parent_connection"):
            return candidate
        root = getattr(candidate, "root", None)
        if root is not None and not _is_robot_annotation(root) and hasattr(root, "parent_connection"):
            return root
    return None
