"""
World serializer — converts a WorldLike (SDT) object into an LLM-readable
text representation.

Extracted and refactored from:
  llmr/pipeline/action_pipeline._serialise_world_for_llm()
  llmr/pipeline/entity_grounder.EntityGrounder._annotation_ground()
  llmr/sdt_interfaces.body_xyz() / body_display_name()

The key design difference: this module is the ONLY place that touches the
world object before handing context to the LLM. The LLM then does all
reasoning — spatial, semantic, and physical — from this serialized view.
No deterministic entity grounding happens here; that is intentionally left
to the LLM.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def serialize_world(world: Any, exec_state: Optional[Any] = None) -> str:
    """
    Serialize the full world state into a human-readable string for LLM context.

    Includes:
    - All scene objects with positions
    - Semantic annotations grouped by body
    - Robot state (arm occupancy) if exec_state is provided

    :param world:      A WorldLike object (SDT World or compatible).
    :param exec_state: Optional ExecutionState carrying held-object info.
    :returns: Multi-line string representation of the world for LLM context.
    """
    sections: List[str] = []

    sections.append(_serialize_bodies(world))
    sections.append(_serialize_annotations(world))

    if exec_state is not None:
        sections.append(_serialize_exec_state(exec_state))

    return "\n\n".join(s for s in sections if s)


# --------------------------------------------------------------------------- #
# Scene bodies                                                                 #
# --------------------------------------------------------------------------- #

def _serialize_bodies(world: Any) -> str:
    bodies = _get_scene_bodies(world)
    if not bodies:
        return ""

    lines = ["=== Scene Objects ==="]
    for body in bodies:
        name = body_display_name(body)
        xyz = body_xyz(body)
        bb = body_bounding_box(body)

        pos_str = (
            f"position=({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})"
            if xyz else "position=unknown"
        )
        dim_str = (
            f"  dims=(w={bb[0]:.2f}, d={bb[1]:.2f}, h={bb[2]:.2f})"
            if bb else ""
        )
        lines.append(f"  - {name}  {pos_str}{dim_str}")

    return "\n".join(lines)


def _get_scene_bodies(world: Any) -> List[Any]:
    """
    Return all non-kinematic scene bodies (filter out robot link bodies).
    Mirrors the filter in llmr/pipeline/action_pipeline._serialise_world_for_llm().
    """
    try:
        return [b for b in world.bodies if _is_scene_body(b)]
    except AttributeError:
        return []


def _is_scene_body(body: Any) -> bool:
    """
    Heuristic: robot kinematic links have slash-separated names (e.g. 'pr2/base_link').
    Scene objects do not.
    """
    try:
        return "/" not in str(body.name)
    except AttributeError:
        return True


# --------------------------------------------------------------------------- #
# Semantic annotations                                                         #
# --------------------------------------------------------------------------- #

def _serialize_annotations(world: Any) -> str:
    annotations = _get_all_annotations(world)
    if not annotations:
        return ""

    lines = ["=== Semantic Annotations ==="]
    for body_name, annos in annotations.items():
        if annos:
            anno_str = ", ".join(type(a).__name__ for a in annos)
            lines.append(f"  - {body_name}: {anno_str}")

    return "\n".join(lines)


def _get_all_annotations(world: Any) -> Dict[str, List[Any]]:
    result: Dict[str, List[Any]] = {}
    try:
        for body in _get_scene_bodies(world):
            name = body_display_name(body)
            try:
                annos = world.get_semantic_annotations_of_body(body)
                result[name] = list(annos) if annos else []
            except Exception:
                result[name] = []
    except AttributeError:
        pass
    return result


# --------------------------------------------------------------------------- #
# Robot / execution state                                                      #
# --------------------------------------------------------------------------- #

def _serialize_exec_state(exec_state: Any) -> str:
    lines = ["=== Robot Arm State ==="]
    try:
        for arm, body in exec_state.held_objects.items():
            if body is not None:
                lines.append(f"  - {arm}: holding {body_display_name(body)}")
            else:
                lines.append(f"  - {arm}: free")
    except AttributeError:
        lines.append("  (arm state unavailable)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Body attribute helpers (public — reusable by other modules)                 #
# These mirror llmr/sdt_interfaces.py helpers but without the SDT import.    #
# --------------------------------------------------------------------------- #

def body_display_name(body: Any) -> str:
    """Return a clean display name for a body, unwrapping PrefixedName chains."""
    try:
        name = body.name
        # Handle PrefixedName or similar wrappers that have a .local attribute
        if hasattr(name, "local"):
            return str(name.local)
        return str(name)
    except AttributeError:
        return repr(body)


def body_xyz(body: Any) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, z) position of a body, or None if unavailable."""
    try:
        pose = body.global_pose
        if pose is None:
            return None
        # Try common pose representations
        if hasattr(pose, "position"):
            p = pose.position
            return (float(p.x), float(p.y), float(p.z))
        if hasattr(pose, "translation"):
            t = pose.translation
            return (float(t[0]), float(t[1]), float(t[2]))
    except Exception:
        pass
    return None


def body_bounding_box(
    body: Any,
    reference_frame: Optional[Any] = None,
) -> Optional[Tuple[float, float, float]]:
    """Return (width, depth, height) bounding box dims, or None if unavailable."""
    try:
        collision = body.collision
        if collision is None:
            return None
        if reference_frame is not None:
            bb = collision.bounding_box(reference_frame)
        else:
            bb = collision.bounding_box()
        if bb is None:
            return None
        # Normalise to (w, d, h) regardless of internal representation
        dims = bb.dimensions if hasattr(bb, "dimensions") else bb
        return (float(dims[0]), float(dims[1]), float(dims[2]))
    except Exception:
        return None
