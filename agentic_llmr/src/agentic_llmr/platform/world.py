"""World — runtime SDT world reference management and scene graph queries.

Merged from: integrations/world_manager.py + resolution/scene.py

Responsibilities:
  - Hold and vend the active SDT world / robot_view references (set by the caller)
  - Answer questions about the live scene: bodies, annotations, poses, relations
  - Provide kinematics helpers used by tool modules

Data access strategy:
  - Bodies and semantic annotations are queried directly through the SDT World
    instance (world.bodies, world.semantic_annotations, world.get_body_by_name).
    The World is the authoritative per-world data store.
  - Forward kinematics, collision geometry, and simulation are world-instance-only
    operations with no KRROOD equivalent.
  - KRROOD (ClassDiagram, Match, EQL) is used in platform/type_bridge.py for
    PyCRAM type introspection and action parameterisation — not here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from krrood.symbol_graph.symbol_graph import SymbolGraph

logger = logging.getLogger(__name__)


# ── Active-world state ─────────────────────────────────────────────────────────

_ACTIVE_WORLD: Any = None
_ACTIVE_ROBOT_VIEW: Any = None

# Callables registered by other modules via register_world_cache().
# All are invoked by set_active_world() so stale caches never outlive a world switch.
_WORLD_CACHE_CLEARERS: List[Callable[[], None]] = []

# Per-world entity caches. The SymbolGraph scan is O(N) over all wrapped
# instances; without caching, every find_body_by_name / get_bodies call repeats
# it. These are populated lazily and cleared on every set_active_world().
_BODIES_CACHE: Optional[List[Any]] = None
_ANNOTATIONS_CACHE: Optional[List[Any]] = None
_BODY_BY_NAME_CACHE: Optional[dict] = None


def register_world_cache(clear_fn: Callable[[], None]) -> None:
    """Register a cache-clear callable to be called on every set_active_world()."""
    _WORLD_CACHE_CLEARERS.append(clear_fn)


def _clear_entity_caches() -> None:
    """Drop the cached body/annotation enumerations (called on world switch)."""
    global _BODIES_CACHE, _ANNOTATIONS_CACHE, _BODY_BY_NAME_CACHE
    _BODIES_CACHE = None
    _ANNOTATIONS_CACHE = None
    _BODY_BY_NAME_CACHE = None


def set_active_world(world: Any, robot_view: Any) -> None:
    """Store references to the active SDT world and robot view, then clear world caches."""
    global _ACTIVE_WORLD, _ACTIVE_ROBOT_VIEW
    _ACTIVE_WORLD = world
    _ACTIVE_ROBOT_VIEW = robot_view
    _clear_entity_caches()
    for clear_fn in _WORLD_CACHE_CLEARERS:
        try:
            clear_fn()
        except Exception as exc:
            logger.debug("world cache clearer failed: %s", exc)


def get_active_world() -> Tuple[Any, Any]:
    """Return (world, robot_view) for the active SDT world.

    Raises:
        RuntimeError: If no world has been initialised yet.
    """
    if _ACTIVE_WORLD is None:
        raise RuntimeError(
            "No active world is set. Call set_active_world() after loading the environment."
        )
    return _ACTIVE_WORLD, _ACTIVE_ROBOT_VIEW


# ── Robot annotation class names ───────────────────────────────────────────────

# Used for MRO-based type discrimination throughout the package.
# A single frozenset here avoids scattered string literals.
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
    """Return (x, y, z) bounding box extents from the body's collision mesh, or None if unavailable.

    Uses body.combined_mesh.extents — pure trimesh computation in body-local frame.
    No world access or FK required.
    reference_frame is accepted for API compatibility but ignored; extents are always
    in body-local frame, which is what all callers need (volume, half-extents for AABB tests).
    """
    try:
        mesh = getattr(body, "combined_mesh", None)
        if mesh is None:
            return None
        ex = mesh.extents  # trimesh: (x_extent, y_extent, z_extent) AABB in local frame
        return float(ex[0]), float(ex[1]), float(ex[2])
    except Exception:
        return None


# ── SymbolGraph entity access ──────────────────────────────────────────────────
#
# SymbolGraph is the primary source for entity enumeration and all property
# queries (names, dimensions, annotation types, etc.).  SDT types are never
# imported here — entities are discriminated by MRO class-name strings so the
# module stays SDT-import-free at the top level.
#
# The SDT World instance is used only where the SymbolGraph has no answer:
#   - forward kinematics  (world.compute_forward_kinematics_np)
#   - simulation context  (world.modify_world, world.root)
# ──────────────────────────────────────────────────────────────────────────────


def get_annotations() -> List[Any]:
    """Return all SemanticAnnotation instances from the SymbolGraph.

    Covers environmental annotations (Milk, Table — have a .bodies attribute)
    and robot-structural annotations (Arm, Gripper — MRO matches
    _SEMANTIC_ANNOTATION_TYPE_NAMES).  Callers can filter with
    _is_robot_annotation() to separate the two groups.
    SDT types are not imported; discrimination is MRO-name-based.
    """
    global _ANNOTATIONS_CACHE
    if _ANNOTATIONS_CACHE is not None:
        return _ANNOTATIONS_CACHE
    try:
        graph = SymbolGraph()
    except Exception as exc:
        logger.debug("get_annotations: SymbolGraph unavailable: %s", exc)
        return []
    result: List[Any] = []
    seen: set = set()
    for wrapped in graph.wrapped_instances:
        inst = wrapped.instance
        if inst is None or id(inst) in seen:
            continue
        seen.add(id(inst))
        mro_names = {cls.__name__ for cls in type(inst).__mro__}
        if hasattr(inst, "bodies") or (mro_names & _SEMANTIC_ANNOTATION_TYPE_NAMES):
            result.append(inst)
    _ANNOTATIONS_CACHE = result
    return result


def get_bodies() -> List[Any]:
    """Return all Body instances from the SymbolGraph (cached per active world).

    Discriminates by MRO class name "Body" — no SDT import needed.
    """
    global _BODIES_CACHE
    if _BODIES_CACHE is not None:
        return _BODIES_CACHE
    try:
        graph = SymbolGraph()
    except Exception as exc:
        logger.debug("get_bodies: SymbolGraph unavailable: %s", exc)
        return []
    result: List[Any] = []
    seen: set = set()
    for wrapped in graph.wrapped_instances:
        inst = wrapped.instance
        if inst is None or id(inst) in seen:
            continue
        seen.add(id(inst))
        if "Body" in {cls.__name__ for cls in type(inst).__mro__}:
            result.append(inst)
    _BODIES_CACHE = result
    return result


# ── Kinematics bridge ──────────────────────────────────────────────────────────


def compute_grasp_descriptions(
    manipulator: Any,
    position_xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    world_root: Any,
) -> Any:
    """Bridge function: construct SDT Pose internally and delegate to GraspDescription.

    Callers pass plain Python tuples — no SDT spatial type imports needed outside this module.
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


def compute_ik_bridge(
    world: Any,
    root_body: Any,
    tip_body: Any,
    x: float, y: float, z: float,
    qx: float, qy: float, qz: float, qw: float,
) -> "Dict[str, float]":
    """Compute IK for root→tip chain toward (x,y,z, qx,qy,qz,qw) in world frame.

    Imports SDT spatial types internally to avoid top-level SDT import.
    Returns {joint_name: position_rad}. Raises on failure (UnreachableException or MaxIterationsException).
    """
    from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
    target = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=x, pos_y=y, pos_z=z,
        quat_x=qx, quat_y=qy, quat_z=qz, quat_w=qw,
        reference_frame=world.root,
    )
    result = world.compute_inverse_kinematics(root=root_body, tip=tip_body, target=target)
    return {
        (str(dof.name.name) if hasattr(dof.name, "name") else str(dof.name)): float(pos)
        for dof, pos in result.items()
    }


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
    """Return the Body whose display name matches *name*, or None if not found.

    Backed by a per-world name→body map (built once from get_bodies()) so repeated
    lookups within a query are O(1) instead of a linear scan each time. The map is
    cleared on every set_active_world().
    """
    global _BODY_BY_NAME_CACHE
    if _BODY_BY_NAME_CACHE is None:
        _BODY_BY_NAME_CACHE = {symbol_display_name(b): b for b in get_bodies()}
    return _BODY_BY_NAME_CACHE.get(name)


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
