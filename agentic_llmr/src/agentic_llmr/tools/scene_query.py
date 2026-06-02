"""Scene query tools — query the semantic scene graph for object poses, relations, and properties.
These tools use the `semantic_digital_twin` package natively.
"""

import functools
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool
from agentic_llmr.tools.artifacts import (
    SceneObjectsArtifact, SemanticAnnotationsArtifact, FindObjectsArtifact,
    ObjectTypeArtifact, ClassifyByRoleArtifact, PoseArtifact, DimensionsArtifact,
    OrientationArtifact, ColorArtifact, SpatialRelationArtifact,
    NearestObjectsArtifact, ObjectsOnSurfaceArtifact, SortBySizeArtifact,
    ArticulatedJointsArtifact, ContainedItemsArtifact,
    JointStatesArtifact, RobotPoseArtifact, EndEffectorPoseArtifact,
    GripperStateArtifact, HeldObjectArtifact, SceneCollisionsArtifact,
    FreeSpotsArtifact, WouldCollideArtifact, IsAccessibleArtifact,
    SupportingObjectArtifact, SupportedByArtifact,
)
from agentic_llmr.platform.world import (
    get_active_world, register_world_cache,
    _is_robot_annotation, sort_by_volume, get_annotations, get_bodies,
    symbol_display_name, get_arm_label, find_body_by_name, get_robot_base_body,
    symbol_bounding_box,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers for Semantic Digital Twin native objects
# ---------------------------------------------------------------------------

def _extract_body_name(body: Any) -> str:
    return symbol_display_name(body) or f"body_{id(body)}"

def _extract_semantic_types(body: Any) -> List[str]:
    body_id = id(body)
    types = []
    for ann in get_annotations():
        try:
            if any(id(b) == body_id for b in getattr(ann, "bodies", [])):
                types.append(ann.__class__.__name__)
        except Exception:
            pass
    return types

def _get_body_position(body: Any) -> Optional[Tuple[float, float, float]]:
    try:
        pose = body.global_pose
        return float(pose.x), float(pose.y), float(pose.z)
    except Exception:
        return None

def _extract_body_pose(body: Any) -> Dict[str, Any]:
    try:
        pose = body.global_pose
        quat = pose.to_quaternion()
        return {
            "position": {"x": float(pose.x), "y": float(pose.y), "z": float(pose.z)},
            "orientation": {"x": float(quat.x), "y": float(quat.y), "z": float(quat.z), "w": float(quat.w)},
        }
    except Exception as e:
        return {"error": f"Could not determine pose: {e}"}

@functools.lru_cache(maxsize=1)
def _build_robot_type_names() -> frozenset:
    names: set = set()
    for ann in get_annotations():
        if _is_robot_annotation(ann):
            for cls in type(ann).__mro__:
                if cls is object:
                    break
                names.add(cls.__name__)
    return frozenset(names)

register_world_cache(_build_robot_type_names.cache_clear)


def _find_annotation_by_query(
    query: str,
    annotations: List[Any],
    *,
    require_attr: Optional[str] = None,
) -> Optional[Any]:
    def _eligible(ann: Any) -> bool:
        return require_attr is None or hasattr(ann, require_attr)

    for ann in annotations:
        if _eligible(ann) and query in ann.__class__.__name__.lower():
            return ann
    for ann in annotations:
        if _eligible(ann) and any(
            _extract_body_name(b).lower() == query for b in getattr(ann, "bodies", [])
        ):
            return ann
    return None


def _resolve_annotation_root(ann: Any) -> Optional[Any]:
    root = getattr(ann, "root", None)
    if root is not None:
        return root
    bodies = getattr(ann, "bodies", [])
    return bodies[0] if bodies else None


def _fmt_pos(
    pos: Optional[Tuple[float, float, float]],
    fallback: str = "unknown",
) -> str:
    if pos is None:
        return fallback
    return f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"


# ---------------------------------------------------------------------------
# Segment 1 — World Inventory & Taxonomy
# ---------------------------------------------------------------------------

class SceneQueryInput(BaseModel):
    pass


class GetSceneObjectsTool(AgenticTool):
    """Return a table of all task-relevant objects with body names, semantic types, and 3D positions."""

    name: str = "list_all_objects"
    description: str = (
        "Scene inventory — returns every task-relevant object in the scene "
        "as a table of body_name, semantic type(s), and 3D position. "
        "A broad starting point when you need an overview of what is present. "
        "Robot structural parts (arms, grippers, base links) are filtered out automatically. "
        "The returned body_name values are the identifiers other tools expect."
    )
    args_schema: Type[BaseModel] = SceneQueryInput

    def _run(self):
        try:
            logger.debug("[SDT Tool] Querying native semantic scene graph...")
            _, robot_view = get_active_world()

            annotations = get_annotations()
            robot_type_names = _build_robot_type_names()

            annotated_ids = set()
            for ann in annotations:
                for b in getattr(ann, "bodies", []):
                    annotated_ids.add(id(b))

            lines = [
                "| body_name | semantic_types | xyz_position |",
                "| --- | --- | --- |"
            ]

            objects = []
            found = 0
            for body in get_bodies():
                if id(body) not in annotated_ids:
                    continue
                name = _extract_body_name(body)
                types = _extract_semantic_types(body)
                task_types = [t for t in types if t not in robot_type_names]
                if not task_types:
                    continue
                types_str = ", ".join(task_types)
                pos = _get_body_position(body)
                xyz = _fmt_pos(pos, fallback="-")
                lines.append(f"| {name} | {types_str} | {xyz} |")
                objects.append({"body_name": name, "types": types_str, "position": xyz})
                found += 1

            robot_base = get_robot_base_body(robot_view)
            if robot_base is not None:
                rname = _extract_body_name(robot_base)
                rpos = _get_body_position(robot_base)
                rxyz = _fmt_pos(rpos, fallback="-")
                lines.append(f"| {rname} | Robot | {rxyz} |")
                objects.append({"body_name": rname, "types": "Robot", "position": rxyz})

            if found == 0:
                lines = ["| annotation | root_body | xyz_position |", "| --- | --- | --- |"]
                for ann in annotations:
                    root = getattr(ann, "root", None)
                    rname = _extract_body_name(root) if root else "-"
                    pos = _get_body_position(root) if root else None
                    xyz = _fmt_pos(pos, fallback="-")
                    lines.append(f"| {ann.__class__.__name__} | {rname} | {xyz} |")
                    objects.append({"body_name": rname, "types": ann.__class__.__name__, "position": xyz})

            artifact = SceneObjectsArtifact(objects=objects)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetSemanticAnnotations
# ---------------------------------------------------------------------------

class SemanticAnnotationsInput(BaseModel):
    pass


class GetSemanticAnnotationsTool(AgenticTool):
    """Return all distinct semantic annotation type names and their associated body names."""

    name: str = "list_object_types"
    description: str = (
        "Type-catalog primitive — lists every semantic annotation class name currently active in the scene "
        "(e.g. Milk, Table, Cereal, Fridge) together with the body names associated with each type. "
        "Use this to discover valid type names when find_objects_by_type returns no match, "
        "or to get a quick inventory of what semantic categories are present. "
        "Does NOT return object positions — use list_all_objects or get_object_pose for that."
    )
    args_schema: Type[BaseModel] = SemanticAnnotationsInput

    def _run(self):
        try:
            logger.debug("[SDT Tool] Listing semantic annotations...")

            summary = {}
            for ann in get_annotations():
                cls_name = ann.__class__.__name__
                if cls_name not in summary:
                    summary[cls_name] = []
                try:
                    for body in ann.bodies:
                        bname = _extract_body_name(body)
                        if bname not in summary[cls_name]:
                            summary[cls_name].append(bname)
                except Exception:
                    pass

            if not summary:
                return "No semantic annotations found in the world.", None

            robot_type_names = _build_robot_type_names()
            lines = ["Active Semantic Annotations:"]
            for cls_name, bnames in summary.items():
                if cls_name in robot_type_names:
                    count = len(bnames)
                    root = bnames[0] if bnames else "-"
                    lines.append(f"- {cls_name}: root={root}  ({count} bodies, robot structural)")
                else:
                    bnames_str = ", ".join(bnames) if bnames else "(no direct bodies)"
                    lines.append(f"- {cls_name}: {bnames_str}")

            artifact = SemanticAnnotationsArtifact(annotations=summary)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: FindObjectsByType
# ---------------------------------------------------------------------------

class FindObjectsByTypeInput(BaseModel):
    semantic_type: str = Field(
        description="Semantic class name to search for (e.g., 'Table', 'Milk', 'Cup'). Case-insensitive partial match."
    )


class FindObjectsByTypeTool(AgenticTool):
    """Resolve a semantic type name to all matching body names and world-frame positions."""

    name: str = "find_objects_by_type"
    description: str = (
        "Resolves a semantic type name to all matching objects with their exact body names and world-frame positions. "
        "Case-insensitive partial match — 'milk' matches 'Milk', 'FreshMilk', etc. "
        "This is how a semantic name (e.g. 'Milk') becomes the exact body_name(s) other tools require. "
        "When no match is found, list_object_types reveals the valid type names."
    )
    args_schema: Type[BaseModel] = FindObjectsByTypeInput

    def _run(self, semantic_type: str):
        try:
            logger.debug(f"[SDT Tool] Searching for semantic type: {semantic_type}")
            query = semantic_type.strip().lower()

            matches = []
            artifact_matches = []
            for ann in get_annotations():
                cls_name = ann.__class__.__name__
                if query not in cls_name.lower():
                    continue
                body_entries = []
                for body in getattr(ann, "bodies", []):
                    bname = _extract_body_name(body)
                    pos = _get_body_position(body)
                    pos_str = _fmt_pos(pos)
                    body_entries.append(f"body_name='{bname}'  pos={pos_str}")
                    pos_dict = {"x": pos[0], "y": pos[1], "z": pos[2]} if pos else {}
                    artifact_matches.append({"cls_name": cls_name, "body_name": bname, "position": pos_dict})
                matches.append((cls_name, body_entries))

            if not matches:
                return (
                    f"No objects matching type '{semantic_type}' found. "
                    "Use list_semantic_annotations to see all available types."
                ), None

            lines = [f"Found {len(matches)} annotation(s) matching '{semantic_type}':"]
            for cls_name, body_entries in matches:
                lines.append(f"  [{cls_name}]")
                for entry in body_entries:
                    lines.append(f"    {entry}")

            artifact = FindObjectsArtifact(type_name=semantic_type, matches=artifact_matches)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetObjectType
# ---------------------------------------------------------------------------

class ObjectTypeInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object (e.g., 'milk.stl').")


class GetObjectTypeTool(AgenticTool):
    """Return the semantic type and full inheritance hierarchy for a specific object."""

    name: str = "get_object_type"
    description: str = (
        "Per-object type query — returns the semantic type and full type hierarchy for a specific object. "
        "Given an exact body_name, returns its direct semantic class (e.g. 'Milk') and the full "
        "inheritance chain (e.g. Milk → FoodItem → Container → SemanticAnnotation). "
        "Use this when you need to know exactly what category a known object belongs to, "
        "or to understand whether an object is a container, surface, food item, etc. "
        "Covers both get_semantic_type and get_semantic_hierarchy in one call."
    )
    args_schema: Type[BaseModel] = ObjectTypeInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Getting semantic type for: {object_name}")

            target_body = find_body_by_name(object_name)
            if target_body is None:
                return f"Object '{object_name}' not found. Use list_all_objects to see valid body names.", None

            target_id = id(target_body)
            matched = []
            for ann in get_annotations():
                for b in getattr(ann, "bodies", []):
                    if id(b) == target_id:
                        direct_type = type(ann).__name__
                        hierarchy = [
                            cls.__name__ for cls in type(ann).__mro__
                            if cls is not object
                        ]
                        matched.append((direct_type, " → ".join(hierarchy), hierarchy))
                        break

            if not matched:
                return f"No semantic annotation found for '{object_name}'.", None

            lines = [f"Object '{object_name}':"]
            all_hierarchies = []
            for direct_type, hierarchy_str, hierarchy in matched:
                lines.append(f"  Semantic type:   {direct_type}")
                lines.append(f"  Type hierarchy:  {hierarchy_str}")
                all_hierarchies.extend(hierarchy)

            artifact = ObjectTypeArtifact(body_name=object_name, type_hierarchy=list(dict.fromkeys(all_hierarchies)))
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: ClassifyObjectsByRole
# ---------------------------------------------------------------------------

class ClassifyByRoleInput(BaseModel):
    pass


class ClassifyObjectsByRoleTool(AgenticTool):
    """Categorise every scene object as surface, articulated fixture, movable object, or robot agent."""

    name: str = "classify_objects_by_role"
    description: str = (
        "Scene-wide structural overview — categorises every object in the scene into one of four roles: "
        "surface (objects you can place things on, e.g. Table, Shelf), "
        "articulated (objects with controllable joints, e.g. Fridge, Drawer, Door), "
        "object (movable items, containers, tools — everything else), "
        "agent (the robot and its structural parts). "
        "A structural map of the scene: what can be opened, what can be placed on, "
        "and what can be picked up."
    )
    args_schema: Type[BaseModel] = ClassifyByRoleInput

    def _run(self):
        try:
            logger.debug("[SDT Tool] Classifying scene objects by role...")
            _, robot_view = get_active_world()

            surfaces = []
            articulateds = []
            objects = []
            agent_entries = []

            art_artifact = []
            surf_artifact = []
            obj_artifact = []
            agent_artifact = []

            for ann in get_annotations():
                if _is_robot_annotation(ann):
                    root = getattr(ann, "root", None)
                    rname = _extract_body_name(root) if root else "-"
                    agent_entries.append((ann.__class__.__name__, rname))
                    agent_artifact.append({"cls_name": ann.__class__.__name__, "body_name": rname})
                    continue

                cls_name = ann.__class__.__name__
                body_names = [_extract_body_name(b) for b in getattr(ann, "bodies", [])]
                primary_name = body_names[0] if body_names else "-"

                pos = None
                root = getattr(ann, "root", None)
                ref_body = root if root is not None else (
                    getattr(ann, "bodies", [None])[0]
                )
                if ref_body is not None:
                    pos = _get_body_position(ref_body)
                pos_str = _fmt_pos(pos)

                if hasattr(ann, "supporting_surface"):
                    surfaces.append((cls_name, primary_name, pos_str))
                    surf_artifact.append({"cls_name": cls_name, "body_name": primary_name, "position": pos_str})
                    continue

                is_articulated = False
                for b in getattr(ann, "bodies", []):
                    conn = getattr(b, "parent_connection", None)
                    if conn is not None and getattr(conn, "is_controlled", False):
                        is_articulated = True
                        break
                if is_articulated:
                    articulateds.append((cls_name, primary_name, pos_str))
                    art_artifact.append({"cls_name": cls_name, "body_name": primary_name, "position": pos_str})
                    continue

                objects.append((cls_name, primary_name, pos_str))
                obj_artifact.append({"cls_name": cls_name, "body_name": primary_name, "position": pos_str})

            lines = ["Scene object roles:"]

            if surfaces:
                lines.append(f"\nSurfaces ({len(surfaces)}):")
                for cls_name, bname, pos_str in surfaces:
                    lines.append(f"  [{cls_name}]  body={bname}  pos={pos_str}")

            if articulateds:
                lines.append(f"\nArticulated ({len(articulateds)}):")
                for cls_name, bname, pos_str in articulateds:
                    lines.append(f"  [{cls_name}]  body={bname}  pos={pos_str}")

            if objects:
                lines.append(f"\nObjects ({len(objects)}):")
                for cls_name, bname, pos_str in objects:
                    lines.append(f"  [{cls_name}]  body={bname}  pos={pos_str}")

            if agent_entries:
                lines.append(f"\nAgent (robot):")
                for cls_name, rname in agent_entries:
                    lines.append(f"  [{cls_name}]  root={rname}")

            if len(lines) == 1:
                return "No objects found in the current scene.", None

            artifact = ClassifyByRoleArtifact(
                surfaces=surf_artifact,
                articulated=art_artifact,
                objects=obj_artifact,
                agent=agent_artifact,
            )
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Segment 2 — Geometric & Spatial Properties
# ---------------------------------------------------------------------------

class ObjectPoseInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object (e.g., 'milk.stl').")

class GetObjectPoseTool(AgenticTool):
    """Return the 6-DoF world-frame pose (position + quaternion) of a named object."""

    name: str = "get_object_pose"
    description: str = "Get the 3D pose (position and orientation) of a specific object by its exact body_name."
    args_schema: Type[BaseModel] = ObjectPoseInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Getting pose for: {object_name}")
            target_body = find_body_by_name(object_name)
            if target_body is None:
                return f"Object '{object_name}' not found in the semantic digital twin.", None
            pose = _extract_body_pose(target_body)
            if "error" in pose:
                return f"Error getting pose for '{object_name}': {pose['error']}", None
            pos = pose["position"]
            ori = pose["orientation"]
            artifact = PoseArtifact(body_name=object_name, position=pos, orientation=ori)
            content = (
                f"Pose of '{object_name}':\n"
                f"  position: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})\n"
                f"  orientation (xyzw): ({ori['x']:.3f}, {ori['y']:.3f}, {ori['z']:.3f}, {ori['w']:.3f})"
            )
            return content, artifact
        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetObjectDimensions
# ---------------------------------------------------------------------------

class ObjectDimensionsInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object (e.g., 'milk.stl').")

class GetObjectDimensionsTool(AgenticTool):
    """Return bounding-box dimensions (width, depth, height) and volume for a named object."""

    name: str = "get_object_dimensions"
    description: str = (
        "Get the physical bounding box (width, depth, height in metres), centre of mass, "
        "and computed volume (width × depth × height in m³) of a specific object. "
        "Essential for grasp approach planning, placement decisions, and size comparisons."
    )
    args_schema: Type[BaseModel] = ObjectDimensionsInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Getting dimensions for: {object_name}")

            target_body = find_body_by_name(object_name)
            if target_body is None:
                return f"Object '{object_name}' not found.", None

            bb_m = {}
            volume_m3 = None
            com = None

            try:
                mesh = getattr(target_body, "combined_mesh", None)
                if mesh is not None:
                    ex = mesh.extents
                    bb_m = {
                        "width":  round(float(ex[0]), 4),
                        "depth":  round(float(ex[1]), 4),
                        "height": round(float(ex[2]), 4),
                    }
                    volume_m3 = round(float(ex[0]) * float(ex[1]) * float(ex[2]), 6)
                else:
                    bb_m = {"error": "No collision mesh available."}
            except Exception:
                bb_m = {"error": "Could not compute bounding box."}

            try:
                com_obj = target_body.center_of_mass
                com = {
                    "x": round(float(com_obj.x), 4),
                    "y": round(float(com_obj.y), 4),
                    "z": round(float(com_obj.z), 4),
                }
            except Exception:
                pos = _get_body_position(target_body)
                if pos:
                    com = {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4),
                           "note": "origin fallback (no mesh COM)"}

            artifact = DimensionsArtifact(
                body_name=object_name,
                bounding_box_m=bb_m,
                volume_m3=volume_m3,
                center_of_mass=com,
            )

            lines = [f"Dimensions of '{object_name}':"]
            if "error" not in bb_m:
                lines.append(f"  bounding_box_m: width={bb_m.get('width')}, depth={bb_m.get('depth')}, height={bb_m.get('height')}")
            else:
                lines.append(f"  bounding_box_m: {bb_m['error']}")
            if volume_m3 is not None:
                lines.append(f"  volume_m3: {volume_m3}")
            if com:
                lines.append(f"  center_of_mass: {com}")

            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetObjectOrientation
# ---------------------------------------------------------------------------

class ObjectOrientationInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object to analyse.")

class GetObjectOrientationTool(AgenticTool):
    """Return orientation status (upright/inverted/lying), tilt angle, and euler angles for a named object."""

    name: str = "get_object_orientation"
    description: str = (
        "Analyse the orientation of an object: whether it is upright, tilted, lying on its side, "
        "or inverted, plus the tilt angle from vertical. Critical for selecting the correct "
        "grasp approach direction."
    )
    args_schema: Type[BaseModel] = ObjectOrientationInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Analysing orientation of: {object_name}")

            body = find_body_by_name(object_name)
            if body is None:
                return f"Object '{object_name}' not found.", None

            T = np.array(body.global_transform.to_np())
            R = T[:3, :3]

            local_z_in_world = R[:, 2]
            world_z = np.array([0.0, 0.0, 1.0])

            cos_angle = float(np.clip(np.dot(local_z_in_world, world_z), -1.0, 1.0))
            tilt_deg = float(np.degrees(np.arccos(cos_angle)))

            if tilt_deg < 20:
                label = "upright (Z-axis pointing up)"
            elif tilt_deg > 160:
                label = "inverted / upside-down"
            elif 70 < tilt_deg < 110:
                label = "lying on its side (Z-axis horizontal)"
            else:
                label = f"tilted at {tilt_deg:.1f}° from vertical"

            roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            pitch = float(np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1))))
            yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
            lz = local_z_in_world.tolist()

            artifact = OrientationArtifact(
                body_name=object_name,
                status=label,
                tilt_deg=round(tilt_deg, 1),
                local_z_in_world=[round(v, 3) for v in lz],
                roll=round(roll, 1),
                pitch=round(pitch, 1),
                yaw=round(yaw, 1),
            )

            content = (
                f"Object '{object_name}' orientation:\n"
                f"  Status: {label}\n"
                f"  Tilt from vertical: {tilt_deg:.1f}°\n"
                f"  Local Z-axis in world: ({lz[0]:.3f}, {lz[1]:.3f}, {lz[2]:.3f})\n"
                f"  RPY (deg): roll={roll:.1f}  pitch={pitch:.1f}  yaw={yaw:.1f}"
            )
            return content, artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Shared helpers for color tools
# ---------------------------------------------------------------------------

def _rgb_to_color_name(r: float, g: float, b: float) -> str:
    if r > 0.7 and g < 0.3 and b < 0.3:
        return "red"
    if r < 0.3 and g > 0.7 and b < 0.3:
        return "green"
    if r < 0.3 and g < 0.3 and b > 0.7:
        return "blue"
    if r > 0.7 and g > 0.7 and b < 0.3:
        return "yellow"
    if r > 0.7 and g > 0.4 and b < 0.2:
        return "orange"
    if r > 0.6 and g < 0.3 and b > 0.6:
        return "purple"
    if r < 0.3 and g > 0.6 and b > 0.6:
        return "cyan"
    if r > 0.8 and g > 0.8 and b > 0.8:
        return "white"
    if r < 0.2 and g < 0.2 and b < 0.2:
        return "black"
    if r > 0.5 and g > 0.5 and b > 0.5:
        return "light grey"
    return f"mixed (R={r:.2f} G={g:.2f} B={b:.2f})"


# ---------------------------------------------------------------------------
# Tool: GetObjectColor
# ---------------------------------------------------------------------------

class ObjectColorInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object.")

class GetObjectColorTool(AgenticTool):
    """Return a human-readable color name derived from the object's geometry."""

    name: str = "get_object_color"
    description: str = (
        "Get the visual color of an object from its geometry. "
        "Use this to disambiguate between multiple objects of the same type "
        "(e.g., 'the red cup' vs 'the blue cup')."
    )
    args_schema: Type[BaseModel] = ObjectColorInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Getting color of: {object_name}")

            body = find_body_by_name(object_name)
            if body is None:
                return f"Object '{object_name}' not found.", None

            shapes = getattr(body, "visual", None)
            if shapes is None:
                return f"'{object_name}' has no visual geometry.", None

            shape_list = getattr(shapes, "shapes", [])
            colors_found = []
            for shape in shape_list:
                color = getattr(shape, "color", None)
                if color is None:
                    continue
                try:
                    rgba = color.to_rgba()
                    r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
                    name = _rgb_to_color_name(r, g, b)
                    colors_found.append(f"{name}  RGBA=({r:.2f}, {g:.2f}, {b:.2f}, {a:.2f})")
                except Exception:
                    continue

            if not colors_found:
                return f"No explicit color found on '{object_name}'. Color may be from a texture or mesh material.", None

            lines = [f"Visual color of '{object_name}':"]
            for c in colors_found:
                lines.append(f"  {c}")

            artifact = ColorArtifact(body_name=object_name, colors=colors_found)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetSpatialRelation
# ---------------------------------------------------------------------------

class SpatialRelationInput(BaseModel):
    object_name: str = Field(description="The body_name of the object to locate.")
    reference_name: str = Field(
        description=(
            "The body_name of the reference object, OR 'robot' to use the robot base. "
            "Example: 'table.urdf', 'milk.stl', 'robot'."
        )
    )

class GetSpatialRelationTool(AgenticTool):
    """Return the direction and Euclidean distance from a reference object to a target object."""

    name: str = "get_spatial_relation"
    description: str = (
        "Compute the spatial relationship between two objects: direction (front/back/left/right/above/below) "
        "and exact distance. Use 'robot' as reference_name to get robot-centric directions. "
        "Essential for resolving instructions like 'the cup to the left of the bowl'."
    )
    args_schema: Type[BaseModel] = SpatialRelationInput

    def _run(self, object_name: str, reference_name: str):
        try:
            logger.debug(f"[SDT Tool] Computing spatial relation: '{object_name}' relative to '{reference_name}'")
            _, robot_view = get_active_world()

            obj_body = find_body_by_name(object_name)
            if obj_body is None:
                return f"Object '{object_name}' not found.", None

            use_robot = reference_name.strip().lower() in ("robot", "me", "self")
            if use_robot:
                ref_body = get_robot_base_body(robot_view)
                ref_label = "robot base"
            else:
                ref_body = find_body_by_name(reference_name)
                ref_label = reference_name

            if ref_body is None:
                return f"Reference '{reference_name}' not found.", None

            obj_pos_raw = _get_body_position(obj_body)
            ref_pos_raw = _get_body_position(ref_body)
            if obj_pos_raw is None:
                return f"Could not compute position for '{object_name}'.", None
            if ref_pos_raw is None:
                return f"Could not compute position for reference '{ref_label}'.", None

            obj_pos = np.array(obj_pos_raw)
            ref_pos = np.array(ref_pos_raw)
            delta = obj_pos - ref_pos
            distance = float(np.linalg.norm(delta))

            try:
                T_ref = np.array(ref_body.global_transform.to_np())
                R_ref = T_ref[:3, :3]
                delta_local = R_ref.T @ delta
                forward_label = "in front of" if delta_local[0] > 0 else "behind"
                lateral_label = "to the left of" if delta_local[1] > 0 else "to the right of"
                vertical_label = "above" if delta_local[2] > 0 else "below"
                dx, dy, dz = delta_local
                frame = f"{ref_label}'s local frame"
            except Exception:
                dx, dy, dz = delta
                forward_label = "+X of" if dx > 0 else "-X of"
                lateral_label = "+Y of" if dy > 0 else "-Y of"
                vertical_label = "above" if dz > 0 else "below"
                frame = "world frame"

            parts = []
            if abs(dz) > 0.05:
                parts.append(f"{abs(dz):.3f}m {vertical_label}")
            if abs(dx) > 0.05:
                parts.append(f"{abs(dx):.3f}m {forward_label}")
            if abs(dy) > 0.05:
                parts.append(f"{abs(dy):.3f}m {lateral_label}")
            direction_str = ", ".join(parts) if parts else "at approximately the same location as"

            content = (
                f"'{object_name}' is {direction_str} '{ref_label}' ({frame}).\n"
                f"Offset vector: dx={dx:.3f}m  dy={dy:.3f}m  dz={dz:.3f}m\n"
                f"Euclidean distance: {distance:.3f}m"
            )
            artifact = SpatialRelationArtifact(
                object_name=object_name,
                reference_name=reference_name,
                direction_str=direction_str,
                offset={"dx": round(float(dx), 3), "dy": round(float(dy), 3), "dz": round(float(dz), 3)},
                distance_m=round(distance, 3),
            )
            return content, artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetNearestObjects
# ---------------------------------------------------------------------------

class NearestObjectsInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the reference object.")
    max_count: int = Field(default=5, description="Maximum number of nearest objects to return (default 5).")
    radius_m: Optional[float] = Field(
        default=None,
        description="If set, only return objects within this Euclidean distance in metres (e.g. 0.5 for 50 cm)."
    )

class GetNearestObjectsTool(AgenticTool):
    """Return the N closest objects to a reference body, sorted by Euclidean distance."""

    name: str = "get_nearest_objects"
    description: str = (
        "Find the N closest semantic objects to a reference object, sorted by Euclidean distance. "
        "Pass radius_m to restrict results to objects within a given distance threshold "
        "(e.g. radius_m=0.5 returns only objects within 50 cm). "
        "max_count caps the list length independently of radius_m. "
        "Useful for spatial disambiguation and building obstacle lists. "
        "Note: results include robot bodies (gripper links, arm links) — "
        "compare_arm_suitability filters robot bodies automatically. "
        "IMPORTANT: object_name must be an exact body_name (e.g. 'milk.stl'). "
        "It does NOT accept semantic class names — use find_objects_by_type first."
    )
    args_schema: Type[BaseModel] = NearestObjectsInput

    def _run(self, object_name: str, max_count: int = 5, radius_m: Optional[float] = None):
        try:
            logger.debug(f"[SDT Tool] Finding nearest objects to: {object_name}")

            ref_body = find_body_by_name(object_name)
            if ref_body is None:
                return f"Reference object '{object_name}' not found.", None

            ref_pos = _get_body_position(ref_body)
            if ref_pos is None:
                return f"Could not compute position for '{object_name}'.", None
            ref_arr = np.array(ref_pos)

            distances = []
            seen_bodies: set = {id(ref_body)}
            for ann in get_annotations():
                for body in getattr(ann, "bodies", []):
                    bid = id(body)
                    if bid in seen_bodies:
                        continue
                    seen_bodies.add(bid)
                    pos = _get_body_position(body)
                    if pos is None:
                        continue
                    dist = float(np.linalg.norm(np.array(pos) - ref_arr))
                    distances.append((dist, ann.__class__.__name__, _extract_body_name(body)))

            if not distances:
                return f"No other annotated objects found to compare with '{object_name}'.", None

            distances.sort(key=lambda x: x[0])
            if radius_m is not None:
                distances = [(d, c, n) for d, c, n in distances if d <= radius_m]
                if not distances:
                    return f"No objects found within {radius_m:.3f}m of '{object_name}'.", None

            selected = distances[:max_count]
            header = f"Nearest objects to '{object_name}'" + (f" (within {radius_m:.3f}m)" if radius_m is not None else "") + ":"
            lines = [header]
            nearest_list = []
            for dist, cls_name, bname in selected:
                lines.append(f"  [{cls_name}]  body: {bname}  distance: {dist:.3f}m")
                nearest_list.append({"cls_name": cls_name, "body_name": bname, "distance_m": round(dist, 3)})

            artifact = NearestObjectsArtifact(reference=object_name, nearest=nearest_list)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetObjectsOnSurface
# ---------------------------------------------------------------------------

class SurfaceQueryInput(BaseModel):
    surface_name: str = Field(
        description=(
            "Semantic type name (e.g., 'Table', 'CounterTop', 'Shelf') or exact body_name "
            "of the surface to query."
        )
    )

class GetObjectsOnSurfaceTool(AgenticTool):
    """Return all objects currently resting on a named surface."""

    name: str = "get_objects_on_surface"
    description: str = (
        "Find all objects currently resting on a given surface (table, shelf, counter, etc.). "
        "Provide the semantic type name or exact body_name of the surface."
    )
    args_schema: Type[BaseModel] = SurfaceQueryInput

    def _run(self, surface_name: str):
        try:
            logger.debug(f"[SDT Tool] Querying objects on surface: {surface_name}")
            query = surface_name.strip().lower()

            annotations = get_annotations()
            surface_ann = _find_annotation_by_query(query, annotations)
            if surface_ann is None:
                return f"No annotation named '{surface_name}' found in this scene.", None

            surface_root = getattr(surface_ann, "root", None)
            surface_pos = _get_body_position(surface_root) if surface_root else None
            if surface_pos is None:
                return f"Objects on {surface_ann.__class__.__name__}: none.", None

            sx, sy, sz = surface_pos
            found = []
            for ann in annotations:
                if ann is surface_ann or _is_robot_annotation(ann):
                    continue
                root = getattr(ann, "root", None)
                if root is None:
                    continue
                pos = _get_body_position(root)
                if pos is None:
                    continue
                dz = pos[2] - sz
                dxy = ((pos[0] - sx) ** 2 + (pos[1] - sy) ** 2) ** 0.5
                if -0.1 <= dz <= 0.6 and dxy <= 2.0:
                    found.append((ann, pos))

            if not found:
                return f"Objects on {surface_ann.__class__.__name__}: none.", None

            lines = [f"Objects on {surface_ann.__class__.__name__}:"]
            obj_list = []
            for ann, pos in found:
                rname = _extract_body_name(getattr(ann, "root", None))
                lines.append(
                    f"  [{type(ann).__name__}]  body={rname}  "
                    f"pos={_fmt_pos(pos)}"
                )
                obj_list.append({
                    "cls_name": type(ann).__name__,
                    "body_name": rname,
                    "position": _fmt_pos(pos),
                })

            artifact = ObjectsOnSurfaceArtifact(surface_name=surface_name, objects=obj_list)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: SortObjectsBySize
# ---------------------------------------------------------------------------

class SortBySizeInput(BaseModel):
    semantic_type: str = Field(
        description="Semantic class name to filter by (e.g., 'Cup', 'Bottle'). Returns all matches sorted by volume."
    )
    largest_first: bool = Field(default=True, description="If true, largest object is listed first.")

class SortObjectsBySizeTool(AgenticTool):
    """Return objects of a semantic type ranked by bounding-box volume."""

    name: str = "sort_objects_by_size"
    description: str = (
        "Find all objects of a given semantic type and rank them by physical volume (largest to smallest). "
        "Use this to resolve instructions like 'the large cup' or 'the small bottle'."
    )
    args_schema: Type[BaseModel] = SortBySizeInput

    def _run(self, semantic_type: str, largest_first: bool = True):
        try:
            logger.debug(f"[SDT Tool] Sorting '{semantic_type}' objects by size...")

            query = semantic_type.strip().lower()
            candidates = [
                ann for ann in get_annotations()
                if query in ann.__class__.__name__.lower() and hasattr(ann, "root")
            ]

            if not candidates:
                return f"No '{semantic_type}' objects with root bodies found in the scene.", None

            sorted_anns = sort_by_volume(candidates, largest_first=largest_first)

            lines = [
                f"'{semantic_type}' objects sorted by size ({'largest' if largest_first else 'smallest'} first):"
            ]
            ranked = []
            for i, ann in enumerate(sorted_anns, 1):
                body_names = [_extract_body_name(b) for b in getattr(ann, "bodies", [])]
                pos = None
                ref_body = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
                if ref_body is not None:
                    pos = _get_body_position(ref_body)
                pos_str = _fmt_pos(pos)
                dims = symbol_bounding_box(ref_body) if ref_body is not None else None
                vol = round(dims[0]*dims[1]*dims[2], 5) if dims else None
                vol_str = f"  vol={vol}m³" if vol else ""
                lines.append(f"  {i}. [{ann.__class__.__name__}]  body: {', '.join(body_names)}  pos: {pos_str}{vol_str}")
                ranked.append({
                    "cls_name": ann.__class__.__name__,
                    "body_names": body_names,
                    "volume_m3": vol,
                    "position": pos_str,
                })

            artifact = SortBySizeArtifact(type_name=semantic_type, ranked=ranked)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Segment 3 — Structural & Topological Relations
# ---------------------------------------------------------------------------

class ArticulatedJointsInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the articulated object (e.g., 'Drawer', 'Fridge', 'Door')."
    )

class GetArticulatedObjectJointsTool(AgenticTool):
    """Return controllable joints of a non-robot articulated object with type, position, and limits."""

    name: str = "get_articulated_object_joints"
    description: str = (
        "Get the controllable joints of a non-robot articulated object (drawer, door, fridge, cabinet, etc.), "
        "including joint type, current position, and limits — what is needed to reason about opening or closing it. "
        "For robot arm joints, use get_joint_states instead."
    )
    args_schema: Type[BaseModel] = ArticulatedJointsInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Getting joints for articulated object: {object_name}")

            query = object_name.strip().lower()
            annotations = get_annotations()
            target_ann = _find_annotation_by_query(query, annotations)
            if target_ann is None:
                return f"No object '{object_name}' found in semantic annotations.", None

            ann_bodies = list(getattr(target_ann, "bodies", []))
            if not ann_bodies:
                return f"'{object_name}' annotation has no associated bodies.", None

            rows = []
            joints_data = []
            seen: set = set()
            for body in ann_bodies:
                conn = getattr(body, "parent_connection", None)
                if conn is None or id(conn) in seen:
                    continue
                seen.add(id(conn))
                if not (hasattr(conn, "position") and hasattr(conn, "dof") and hasattr(conn, "is_controlled")):
                    continue
                try:
                    name_obj = getattr(conn, "name", None)
                    jname = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
                    conn_type = type(conn).__name__
                    jtype = "revolute" if conn_type == "RevoluteConnection" else (
                        "prismatic" if conn_type == "PrismaticConnection" else "active1dof"
                    )
                    pos = round(float(conn.position), 5)
                    dof = conn.dof
                    lo, hi = None, None
                    if dof.has_position_limits():
                        lims = dof.limits
                        lo = float(lims.lower.position) if lims.lower is not None else None
                        hi = float(lims.upper.position) if lims.upper is not None else None
                        lim_str = f"[{lo:.3f}, {hi:.3f}]" if (lo is not None and hi is not None) else "limited"
                    else:
                        lim_str = "unlimited"
                    rows.append(f"  {jname}  type={jtype}  pos={pos}  limits={lim_str}")
                    joints_data.append({"name": jname, "type": jtype, "position": pos, "limits": [lo, hi]})
                except Exception:
                    continue

            if not rows:
                return (
                    f"No active controllable joints found for '{target_ann.__class__.__name__}'. "
                    "It may be a rigid object."
                ), None

            lines = [f"Joints of [{target_ann.__class__.__name__}]:"]
            lines.extend(rows)
            artifact = ArticulatedJointsArtifact(
                object_name=object_name,
                cls_name=target_ann.__class__.__name__,
                joints=joints_data,
            )
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetContainedItems
# ---------------------------------------------------------------------------

class ContainedItemsInput(BaseModel):
    container_name: str = Field(
        description=(
            "Semantic type name or body_name of the container "
            "(e.g., 'Fridge', 'Drawer', 'Bowl', 'fridge.urdf')."
        )
    )

def _lookup_semantic_contents(container_ann: Any) -> Optional[List[Tuple]]:
    semantic_objects = getattr(container_ann, "objects", None)
    if not semantic_objects:
        return None
    found = []
    for obj in semantic_objects:
        if _is_robot_annotation(obj):
            continue
        ref = _resolve_annotation_root(obj)
        if ref is None:
            continue
        pos = _get_body_position(ref)
        found.append((obj.__class__.__name__, _extract_body_name(ref), pos))
    return found if found else None


def _detect_geometric_contents(
    container_ann: Any,
    container_root: Any,
    T_c: np.ndarray,
    dims: Tuple[float, float, float],
    annotations: List[Any],
) -> List[Tuple]:
    t_c = T_c[:3, 3]
    R_c = T_c[:3, :3]
    depth, width, height = dims
    container_body_ids = {id(b) for b in getattr(container_ann, "bodies", [])}

    xy_margin = 0.05
    z_lo = -(height / 2) - xy_margin
    z_hi = height + xy_margin

    found = []
    for ann in annotations:
        if ann is container_ann or _is_robot_annotation(ann):
            continue
        ref = _resolve_annotation_root(ann)
        if ref is None or id(ref) in container_body_ids:
            continue
        pos_world = _get_body_position(ref)
        if pos_world is None:
            continue
        p_local = R_c.T @ (np.array(pos_world) - t_c)
        if (
            abs(p_local[0]) <= depth / 2 + xy_margin
            and abs(p_local[1]) <= width / 2 + xy_margin
            and z_lo <= p_local[2] <= z_hi
        ):
            found.append((ann.__class__.__name__, _extract_body_name(ref), tuple(pos_world)))
    return found


class GetContainedItemsTool(AgenticTool):
    """Return objects inside a container via semantic list or geometric AABB containment fallback."""

    name: str = "get_contained_items"
    description: str = (
        "Returns all objects whose positions fall within the 3D spatial volume of a container "
        "(fridge interior, drawer cavity, bowl, box, etc.). "
        "Uses geometric containment detection in the container's local frame — independent of "
        "whether the container is open or closed. "
        "To check whether contained items are actually reachable, also query "
        "get_articulated_object_joints for the container's joint state."
    )
    args_schema: Type[BaseModel] = ContainedItemsInput

    def _run(self, container_name: str):
        try:
            logger.debug(f"[SDT Tool] Finding items contained in: {container_name}")
            query = container_name.strip().lower()
            annotations = get_annotations()

            container_ann = _find_annotation_by_query(query, annotations)
            if container_ann is None:
                return f"No container named '{container_name}' found in the scene.", None

            container_root = _resolve_annotation_root(container_ann)
            if container_root is None:
                return f"Container '{container_ann.__class__.__name__}' has no root body.", None

            T_c = np.array(container_root.global_transform.to_np())
            dims = symbol_bounding_box(container_root)
            if dims is None:
                return (
                    f"Could not compute bounding box for '{container_ann.__class__.__name__}'. "
                    "Containment check unavailable."
                ), None

            found = _lookup_semantic_contents(container_ann)
            method = "semantic"
            if found is None:
                found = _detect_geometric_contents(container_ann, container_root, T_c, dims, annotations)
                method = "geometric"

            if not found:
                return f"No objects found inside '{container_ann.__class__.__name__}'.", None

            lines = [
                f"Objects inside '{container_ann.__class__.__name__}' "
                f"({len(found)} found, {method} detection):"
            ]
            items_list = []
            for cls_name, bname, pos in found:
                pos_str = _fmt_pos(pos)
                lines.append(f"  [{cls_name}]  body={bname}  pos={pos_str}")
                items_list.append({"cls_name": cls_name, "body_name": bname, "position": pos_str})

            artifact = ContainedItemsArtifact(
                container=container_name,
                items=items_list,
                method=method,
            )
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Segment 4 — Functional State & Affordances (stubs)
# ---------------------------------------------------------------------------

class ObjectStateInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object (e.g., 'Milk', 'milk.stl')."
    )

class GetObjectStateTool(AgenticTool):
    """Stub — returns a not-implemented message until SDT models dynamic object state."""

    name: str = "get_object_state"
    description: str = (
        "Returns the dynamic functional state of an object: temperature, fill level, "
        "cleanliness, power state, etc. "
        "[NOT YET IMPLEMENTED — requires SDT dynamic state modeling]"
    )
    args_schema: Type[BaseModel] = ObjectStateInput

    def _run(self, object_name: str):
        msg = (
            "get_object_state is not yet implemented. "
            "The SDT framework does not currently model dynamic object state "
            "(temperature, fill_level, is_clean, is_powered, etc.). "
            "This tool will be completed once those attributes are added to the annotation classes."
        )
        return msg, None


class ObjectAffordancesInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object (e.g., 'Fridge', 'cup.stl')."
    )

class GetObjectAffordancesTool(AgenticTool):
    """Stub — returns a not-implemented message until SDT models object affordances."""

    name: str = "get_object_affordances"
    description: str = (
        "Returns a structured affordance profile for an object: whether it can be picked up, "
        "opened, filled, placed on, or used as a container. "
        "[PARTIAL STUB — type-hierarchy affordances derivable now; "
        "is_pickable requires Segment 7 reachability]"
    )
    args_schema: Type[BaseModel] = ObjectAffordancesInput

    def _run(self, object_name: str):
        msg = (
            "get_object_affordances is not yet fully implemented. "
            "In the meantime, derive affordances from: "
            "(1) get_object_type — check MRO for HasSupportingSurface, HasStorageSpace, "
            "HasDoors, HasDrawers, HasHandle, HasHinge, HasSlider; "
            "(2) get_articulated_object_joints — confirms openable joints; "
            "(3) classify_objects_by_role — confirms surface/articulated/object role."
        )
        return msg, None


# ---------------------------------------------------------------------------
# Segment 5 — Robot & Interaction State
# ---------------------------------------------------------------------------

class JointStatesInput(BaseModel):
    pass

class GetJointStatesTool(AgenticTool):
    """Return current position and limits for all active robot arm joints."""

    name: str = "get_joint_states"
    description: str = (
        "Get the current position of all active robot arm joints (shoulder, elbow, wrist, etc.). "
        "Returns only the robot's own controllable arm joints — not joints of furniture or objects. "
        "Use this to inspect the robot's current arm configuration (e.g. parked vs extended). "
        "This reports current positions only; for a joint's motion limits use the kinematics "
        "specialist's joint-limit capability. For joints of non-robot articulated objects like "
        "drawers or doors, use get_articulated_object_joints instead."
    )
    args_schema: Type[BaseModel] = JointStatesInput

    def _run(self):
        try:
            logger.debug("[SDT Tool] Reading active joint states...")
            _, robot_view = get_active_world()

            rows = []
            joints_data = []
            parts = []
            for attr in ("arms", "neck", "torso", "base", "drive"):
                val = getattr(robot_view, attr, None)
                if val is None:
                    continue
                parts.extend(val if isinstance(val, list) else [val])

            seen = set()
            for part in parts:
                for js in getattr(part, "joint_states", []):
                    for conn in getattr(js, "connections", []):
                        conn_id = id(conn)
                        if conn_id in seen:
                            continue
                        seen.add(conn_id)
                        if not (hasattr(conn, "position") and hasattr(conn, "dof") and hasattr(conn, "is_controlled")):
                            continue
                        if not getattr(conn, "is_controlled", False):
                            continue
                        try:
                            name_obj = getattr(conn, "name", None)
                            name = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
                            pos = round(float(conn.position), 5)
                            rows.append(f"  {name}: position={pos}")
                            joints_data.append({"name": name, "position": pos})
                        except Exception:
                            continue

            if not rows:
                return "No controllable joints found via robot model traversal.", None

            artifact = JointStatesArtifact(joints=joints_data)
            return "Active Joint States:\n" + "\n".join(rows), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetRobotPose
# ---------------------------------------------------------------------------

class RobotPoseInput(BaseModel):
    pass

class GetRobotPoseTool(AgenticTool):
    """Return the 6-DoF world-frame pose of the robot base link."""

    name: str = "get_robot_pose"
    description: str = (
        "Get the current world-frame position and orientation of the robot base. "
        "Use this as a reference for distance calculations or spatial reasoning "
        "relative to the robot. Returns position (x, y, z) and orientation as quaternion."
    )
    args_schema: Type[BaseModel] = RobotPoseInput

    def _run(self):
        try:
            logger.debug("[SDT Tool] Getting robot base pose...")
            _, robot_view = get_active_world()
            from scipy.spatial.transform import Rotation as R

            base_body = get_robot_base_body(robot_view)
            if base_body is None:
                return "Could not locate robot base body.", None

            T = np.array(base_body.global_transform.to_np())
            pos = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
            bname = _extract_body_name(base_body)

            artifact = RobotPoseArtifact(
                body_name=bname,
                position=[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                orientation=[round(float(quat[0]), 3), round(float(quat[1]), 3),
                             round(float(quat[2]), 3), round(float(quat[3]), 3)],
            )
            content = (
                f"Robot base ({bname}):\n"
                f"  position: ({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f})\n"
                f"  orientation (xyzw): ({float(quat[0]):.3f}, {float(quat[1]):.3f}, "
                f"{float(quat[2]):.3f}, {float(quat[3]):.3f})"
            )
            return content, artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetEndEffectorPose
# ---------------------------------------------------------------------------

class EndEffectorPoseInput(BaseModel):
    arm: str = Field(description="Which arm to query: 'left' or 'right'.")

class GetEndEffectorPoseTool(AgenticTool):
    """Return the current world-frame pose of a gripper tool frame by arm name."""

    name: str = "get_end_effector_pose"
    description: str = (
        "Get the current world-frame pose of the robot's gripper tool frame for a given arm. "
        "Use this to know where the gripper is right now — useful before grasping, "
        "for proximity checks, or to understand the current arm configuration geometrically."
    )
    args_schema: Type[BaseModel] = EndEffectorPoseInput

    def _run(self, arm: str):
        try:
            logger.debug(f"[SDT Tool] Getting end-effector pose for: {arm} arm")
            _, robot_view = get_active_world()
            from scipy.spatial.transform import Rotation as R

            side = arm.strip().lower()
            target_arm = None
            for a in getattr(robot_view, "arms", []):
                if get_arm_label(a, robot_view) == side:
                    target_arm = a
                    break

            if target_arm is None:
                return f"No arm matching '{arm}' found. Use 'left' or 'right'.", None

            manip = getattr(target_arm, "manipulator", None)
            tool_frame = getattr(manip, "tool_frame", None) if manip else None
            if tool_frame is None:
                tool_frame = getattr(target_arm, "tip", None)
            if tool_frame is None:
                return f"Could not find tool frame for {arm} arm.", None

            T = np.array(tool_frame.global_transform.to_np())
            pos = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()
            tf_name = _extract_body_name(tool_frame)

            artifact = EndEffectorPoseArtifact(
                arm=side,
                body_name=tf_name,
                position=[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
                orientation=[round(float(quat[0]), 3), round(float(quat[1]), 3),
                             round(float(quat[2]), 3), round(float(quat[3]), 3)],
            )
            content = (
                f"{side.upper()} end-effector ({tf_name}):\n"
                f"  position: ({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f})\n"
                f"  orientation (xyzw): ({float(quat[0]):.3f}, {float(quat[1]):.3f}, "
                f"{float(quat[2]):.3f}, {float(quat[3]):.3f})"
            )
            return content, artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetGripperState
# ---------------------------------------------------------------------------

class GripperStateInput(BaseModel):
    arm: str = Field(description="Which arm's gripper to query: 'left' or 'right'.")

class GetGripperStateTool(AgenticTool):
    """Return the opening width and open/closed/partial status of a named gripper."""

    name: str = "get_gripper_state"
    description: str = (
        "Get the current opening state of the robot's gripper for a given arm. "
        "Returns whether the gripper is open, closed, or partially open, "
        "and the approximate opening width in metres. "
        "To determine what object is being held, use get_held_object instead."
    )
    args_schema: Type[BaseModel] = GripperStateInput

    def _run(self, arm: str):
        try:
            logger.debug(f"[SDT Tool] Getting gripper state for: {arm} arm")
            _, robot_view = get_active_world()

            side = arm.strip().lower()
            target_arm = None
            for a in getattr(robot_view, "arms", []):
                if get_arm_label(a, robot_view) == side:
                    target_arm = a
                    break

            if target_arm is None:
                return f"No arm matching '{arm}' found. Use 'left' or 'right'.", None

            manip = getattr(target_arm, "manipulator", None)
            if manip is None:
                return f"No gripper/manipulator attached to {arm} arm.", None

            gripper_type = type(manip).__name__

            width_m = None
            if hasattr(manip, "finger") and hasattr(manip, "thumb"):
                finger = getattr(manip, "finger", None)
                thumb = getattr(manip, "thumb", None)
                if finger is not None and thumb is not None:
                    try:
                        T_f = np.array(finger.tip.global_transform.to_np())
                        T_t = np.array(thumb.tip.global_transform.to_np())
                        width_m = float(np.linalg.norm(T_f[:3, 3] - T_t[:3, 3]))
                    except Exception:
                        pass
            elif hasattr(manip, "fingers"):
                fingers = getattr(manip, "fingers", [])
                thumb = getattr(manip, "thumb", None)
                if fingers and thumb is not None:
                    try:
                        T_f = np.array(fingers[0].tip.global_transform.to_np())
                        T_t = np.array(thumb.tip.global_transform.to_np())
                        width_m = float(np.linalg.norm(T_f[:3, 3] - T_t[:3, 3]))
                    except Exception:
                        pass

            pct = None
            seen: set = set()
            for js in getattr(manip, "joint_states", []):
                for conn in getattr(js, "connections", []):
                    if id(conn) in seen:
                        continue
                    seen.add(id(conn))
                    if not (hasattr(conn, "position") and hasattr(conn, "dof") and hasattr(conn, "is_controlled")):
                        continue
                    if not getattr(conn, "is_controlled", False):
                        continue
                    dof = conn.dof
                    if dof.has_position_limits():
                        lo = float(dof.limits.lower.position)
                        hi = float(dof.limits.upper.position)
                        if hi > lo:
                            pct = round((float(conn.position) - lo) / (hi - lo) * 100, 1)
                    break
                if pct is not None:
                    break

            if width_m is None and pct is None:
                return f"Could not determine gripper state for {side} arm ({gripper_type}).", None

            if pct is not None:
                status = "closed" if pct < 5 else ("fully open" if pct > 90 else "partially open")
            else:
                status = "closed" if width_m < 0.005 else ("fully open" if width_m > 0.07 else "partially open")

            parts = [f"{side.upper()} gripper ({gripper_type}): {status}"]
            if width_m is not None:
                parts.append(f"  finger-to-thumb gap: {width_m:.4f}m")
            if pct is not None:
                parts.append(f"  joint range used: {pct:.1f}%")

            artifact = GripperStateArtifact(
                arm=side,
                gripper_type=gripper_type,
                status=status,
                width_m=round(width_m, 4) if width_m is not None else None,
                range_pct=pct,
            )
            return "\n".join(parts), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetHeldObject
# ---------------------------------------------------------------------------

class HeldObjectInput(BaseModel):
    arm: str = Field(description="Which arm to query: 'left' or 'right'.")

class GetHeldObjectTool(AgenticTool):
    """Return the object currently grasped by a given arm, detected via kinematic re-parenting."""

    name: str = "get_held_object"
    description: str = (
        "Returns the object currently held in the specified arm's gripper, or reports "
        "that the arm is empty. "
        "Detects grasped objects by inspecting the kinematic tree: when PyCRAM picks up "
        "an object it re-parents the object's root body to the gripper via a fixed "
        "connection — this tool finds that attachment. "
        "get_gripper_state only tells you the opening width; "
        "this tool tells you WHAT is being held."
    )
    args_schema: Type[BaseModel] = HeldObjectInput

    def _run(self, arm: str):
        try:
            logger.debug(f"[SDT Tool] Checking held object for: {arm} arm")
            _, robot_view = get_active_world()
            side = arm.strip().lower()

            target_arm = None
            for a in getattr(robot_view, "arms", []):
                if get_arm_label(a, robot_view) == side:
                    target_arm = a
                    break
            if target_arm is None:
                return f"No arm matching '{arm}' found. Use 'left' or 'right'.", None

            gripper_ids = _collect_gripper_body_ids(target_arm)
            if not gripper_ids:
                return f"Could not resolve gripper bodies for {side} arm.", None

            for ann in get_annotations():
                if _is_robot_annotation(ann):
                    continue
                root = getattr(ann, "root", None)
                if root is None:
                    continue
                conn = getattr(root, "parent_connection", None)
                if conn is None:
                    continue
                if hasattr(conn, "is_controlled"):
                    continue
                parent_body = getattr(conn, "parent", None)
                if parent_body is not None and id(parent_body) in gripper_ids:
                    bname = _extract_body_name(root)
                    pos = _get_body_position(root)
                    pos_str = _fmt_pos(pos)
                    artifact = HeldObjectArtifact(
                        arm=side,
                        body_name=bname,
                        cls_name=ann.__class__.__name__,
                        position=[round(float(p), 3) for p in pos] if pos else None,
                    )
                    return (
                        f"{side.upper()} arm is holding: [{ann.__class__.__name__}]  "
                        f"body={bname}  pos={pos_str}"
                    ), artifact

            artifact = HeldObjectArtifact(arm=side, body_name=None, cls_name=None, position=None)
            return f"{side.upper()} arm is not holding any object.", artifact

        except Exception as e:
            return self._handle_error(e), None


def _collect_gripper_body_ids(arm: Any) -> set:
    ids: set = set()

    tip = getattr(arm, "tip", None)
    if tip is not None:
        ids.add(id(tip))

    manip = getattr(arm, "manipulator", None)
    if manip is None:
        return ids

    tool_frame = getattr(manip, "tool_frame", None)
    if tool_frame is not None:
        ids.add(id(tool_frame))

    for part_attr in ("finger", "thumb"):
        part = getattr(manip, part_attr, None)
        if part is None:
            continue
        for body_attr in ("root", "tip"):
            b = getattr(part, body_attr, None)
            if b is not None:
                ids.add(id(b))

    for f in getattr(manip, "fingers", []):
        for body_attr in ("root", "tip"):
            b = getattr(f, body_attr, None)
            if b is not None:
                ids.add(id(b))

    for js in getattr(manip, "joint_states", []):
        for conn in getattr(js, "connections", []):
            for body_attr in ("parent", "child"):
                b = getattr(conn, body_attr, None)
                if b is not None:
                    ids.add(id(b))

    return ids


# ---------------------------------------------------------------------------
# Segment 6 — Collision, Free Space & Placement
# ---------------------------------------------------------------------------

class SceneCollisionsInput(BaseModel):
    pass

class CheckSceneCollisionsTool(AgenticTool):
    """Return all current body-body collisions and penetration distances in the scene."""

    name: str = "check_scene_collisions"
    description: str = (
        "Check whether any objects in the current scene are in collision. "
        "Returns all colliding body pairs and their penetration distances. "
        "Use this to verify a configuration is collision-free before executing."
    )
    args_schema: Type[BaseModel] = SceneCollisionsInput

    def _run(self):
        try:
            logger.debug("[SDT Tool] Checking scene collisions...")
            world, _ = get_active_world()

            result = world.collision_manager.compute_collisions()

            if not result.any():
                artifact = SceneCollisionsArtifact(any_collision=False, contacts=[])
                return "No collisions detected in the current scene.", artifact

            lines = [f"Collisions detected ({len(result.contacts)} contact(s)):"]
            contacts = []
            for contact in result.contacts:
                name_a = _extract_body_name(contact.body_a)
                name_b = _extract_body_name(contact.body_b)
                dist = round(float(contact.distance), 5)
                status = "PENETRATING" if dist < 0 else "touching"
                lines.append(f"  {name_a}  <-->  {name_b}  distance={dist}m  [{status}]")
                contacts.append({"body_a": name_a, "body_b": name_b, "distance_m": dist})

            artifact = SceneCollisionsArtifact(any_collision=True, contacts=contacts)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetFreePlacementSpots
# ---------------------------------------------------------------------------

class FreePlacementSpotsInput(BaseModel):
    surface_name: str = Field(
        description="Semantic type name or body_name of the surface (e.g., 'Table', 'Shelf')."
    )
    count: int = Field(default=5, description="Number of candidate placement positions to return.")
    check_occupancy: bool = Field(
        default=True,
        description=(
            "If True (default), filter out positions whose footprint overlaps with objects "
            "already on the surface. Set to False to return the raw geometric grid."
        ),
    )
    object_footprint_m: float = Field(
        default=0.1,
        description=(
            "Half-width (metres) of the object to be placed, used as clearance radius "
            "when filtering occupied spots. Default 0.1 m (10 cm)."
        ),
    )

def _find_surface_annotation(query: str) -> Optional[Any]:
    return _find_annotation_by_query(query, get_annotations(), require_attr="supporting_surface")


def _compute_surface_geometry(root_body: Any) -> Tuple[float, float, float, float, float]:
    surface_pose = root_body.global_pose
    cx, cy, cz = float(surface_pose.x), float(surface_pose.y), float(surface_pose.z)
    mesh = getattr(root_body, "combined_mesh", None)
    if mesh is not None:
        ex, ey, ez = [float(v) for v in mesh.extents]
        return cx, cy, cz + ez * 0.5, ex * 0.4, ey * 0.4
    return cx, cy, cz + 0.02, 0.3, 0.3


def _build_placement_grid(
    cx: float, cy: float, z_top: float, half_x: float, half_y: float, count: int
) -> List[Tuple[float, float, float]]:
    import numpy as _np
    grid_n = max(4, int((count * 4) ** 0.5) + 1)
    xs = _np.linspace(cx - half_x, cx + half_x, grid_n)
    ys = _np.linspace(cy - half_y, cy + half_y, grid_n)
    return [(float(x), float(y), z_top) for x in xs for y in ys]


def _collect_occupied_footprints(
    surface_ann: Any, z_top: float
) -> List[Tuple[float, float, float, float]]:
    occupied = []
    for ann in get_annotations():
        if _is_robot_annotation(ann) or ann is surface_ann:
            continue
        ref = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
        if ref is None:
            continue
        pos = _get_body_position(ref)
        if pos is None or abs(pos[2] - z_top) > 0.25:
            continue
        dims = symbol_bounding_box(ref)
        hw = (dims[1] / 2.0) if dims else 0.1
        hd = (dims[0] / 2.0) if dims else 0.1
        occupied.append((float(pos[0]), float(pos[1]), hw, hd))
    return occupied


def _filter_free_spots(
    candidates: List[Tuple[float, float, float]],
    occupied: List[Tuple[float, float, float, float]],
    clearance: float,
) -> List[Tuple[float, float, float]]:
    free = []
    for cx_c, cy_c, cz_c in candidates:
        blocked = any(
            abs(cx_c - ox) < hw + clearance and abs(cy_c - oy) < hd + clearance
            for ox, oy, hw, hd in occupied
        )
        if not blocked:
            free.append((cx_c, cy_c, cz_c))
    return free


class GetFreePlacementSpotsTool(AgenticTool):
    """Return a grid of collision-free placement positions on a named surface."""

    name: str = "get_free_placement_spots"
    description: str = (
        "Sample candidate placement positions on a surface. Returns world-frame (x, y, z) positions "
        "that are on the surface and clear of existing objects. "
        "Set check_occupancy=True (default) to filter out spots already occupied by other objects; "
        "supply object_footprint_m (half-width in metres, default 0.1 m) to match the object being placed. "
        "Use this before a PlaceAction to find a valid target pose."
    )
    args_schema: Type[BaseModel] = FreePlacementSpotsInput

    def _run(
        self,
        surface_name: str,
        count: int = 5,
        check_occupancy: bool = True,
        object_footprint_m: float = 0.1,
    ):
        try:
            logger.debug(f"[SDT Tool] Sampling placement spots on: {surface_name}")

            query = surface_name.strip().lower()
            surface_ann = _find_surface_annotation(query)
            if surface_ann is None:
                return f"No annotation named '{surface_name}' with placement support exists in this scene.", None

            root_body = getattr(surface_ann, "root", None)
            if root_body is None:
                return f"Surface '{surface_name}' has no root body.", None

            cx, cy, z_top, half_x, half_y = _compute_surface_geometry(root_body)
            candidates = _build_placement_grid(cx, cy, z_top, half_x, half_y, count)

            if check_occupancy:
                occupied = _collect_occupied_footprints(surface_ann, z_top)
                candidates = _filter_free_spots(candidates, occupied, object_footprint_m)

            selected = candidates[:count]
            if not selected:
                return (
                    f"No free placement spots found on [{surface_ann.__class__.__name__}] "
                    f"with footprint clearance {object_footprint_m:.2f} m. "
                    "Try a smaller object_footprint_m or check_occupancy=False."
                ), None

            filtered_note = " (occupancy-filtered)" if check_occupancy else " (raw grid)"
            lines = [
                f"Candidate placement positions on [{surface_ann.__class__.__name__}]{filtered_note}:"
            ]
            spots = []
            for i, (x, y, z) in enumerate(selected, 1):
                lines.append(f"  {i}. position: ({x:.3f}, {y:.3f}, {z:.3f})")
                spots.append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})

            artifact = FreeSpotsArtifact(
                surface_name=surface_name,
                cls_name=surface_ann.__class__.__name__,
                spots=spots,
                occupancy_filtered=check_occupancy,
            )
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: WouldCollideAtPose
# ---------------------------------------------------------------------------

class WouldCollideAtPoseInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object to test (e.g., 'Milk', 'milk.stl')."
    )
    x: float = Field(description="Target world-frame X position in metres.")
    y: float = Field(description="Target world-frame Y position in metres.")
    z: float = Field(description="Target world-frame Z position in metres.")
    clearance_m: float = Field(
        default=0.02,
        description=(
            "Extra safety margin added to each object's half-extents during the overlap test. "
            "Default 0.02 m (2 cm)."
        ),
    )

class WouldCollideAtPoseTool(AgenticTool):
    """Check whether placing an object at a given (x, y, z) would cause an AABB overlap."""

    name: str = "would_collide_at_pose"
    description: str = (
        "Test whether placing an object at a target (x, y, z) position would cause its "
        "axis-aligned bounding box to overlap with any other object currently in the scene. "
        "Returns True/False plus a list of conflicting objects. "
        "Uses conservative AABB intersection — no world mutation required. "
        "Use this to validate a candidate placement position before executing a PlaceAction, "
        "or to choose between multiple candidates from get_free_placement_spots."
    )
    args_schema: Type[BaseModel] = WouldCollideAtPoseInput

    def _run(
        self,
        object_name: str,
        x: float,
        y: float,
        z: float,
        clearance_m: float = 0.02,
    ):
        try:
            logger.debug(f"[SDT Tool] Checking hypothetical placement of '{object_name}' at ({x:.3f},{y:.3f},{z:.3f})")

            obj_body = find_body_by_name(object_name)
            if obj_body is None:
                for ann in get_annotations():
                    if object_name.strip().lower() in ann.__class__.__name__.lower():
                        obj_body = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
                        break
            if obj_body is None:
                return f"Object '{object_name}' not found in the scene.", None

            dims = symbol_bounding_box(obj_body)
            if dims:
                obj_hd = dims[0] / 2.0 + clearance_m
                obj_hw = dims[1] / 2.0 + clearance_m
                obj_hh = dims[2] / 2.0 + clearance_m
            else:
                mesh = getattr(obj_body, "combined_mesh", None)
                if mesh is not None:
                    ex, ey, ez = [float(v) / 2.0 + clearance_m for v in mesh.extents]
                    obj_hd, obj_hw, obj_hh = ex, ey, ez
                else:
                    obj_hd = obj_hw = obj_hh = 0.1 + clearance_m

            obj_ann_id = None
            for ann in get_annotations():
                bodies = getattr(ann, "bodies", [])
                root = getattr(ann, "root", None)
                if root is obj_body or obj_body in bodies:
                    obj_ann_id = id(ann)
                    break

            conflicts_list: List[str] = []

            for ann in get_annotations():
                if id(ann) == obj_ann_id:
                    continue
                if _is_robot_annotation(ann):
                    continue
                ref = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
                if ref is None:
                    continue
                other_pos = _get_body_position(ref)
                if other_pos is None:
                    continue

                other_dims = symbol_bounding_box(ref)
                if other_dims:
                    other_hd = other_dims[0] / 2.0
                    other_hw = other_dims[1] / 2.0
                    other_hh = other_dims[2] / 2.0
                else:
                    other_mesh = getattr(ref, "combined_mesh", None)
                    if other_mesh is not None:
                        other_hd, other_hw, other_hh = [float(v) / 2.0 for v in other_mesh.extents]
                    else:
                        other_hd = other_hw = other_hh = 0.05

                ox, oy, oz = float(other_pos[0]), float(other_pos[1]), float(other_pos[2])
                overlap_x = abs(x - ox) < (obj_hd + other_hd)
                overlap_y = abs(y - oy) < (obj_hw + other_hw)
                overlap_z = abs(z - oz) < (obj_hh + other_hh)

                if overlap_x and overlap_y and overlap_z:
                    bname = _extract_body_name(ref)
                    conflicts_list.append(f"[{ann.__class__.__name__}] body={bname}")

            artifact = WouldCollideArtifact(
                object_name=object_name,
                x=x, y=y, z=z,
                collides=len(conflicts_list) > 0,
                conflicts=conflicts_list,
            )

            if not conflicts_list:
                return (
                    f"No collision: '{object_name}' can be placed at ({x:.3f}, {y:.3f}, {z:.3f}) "
                    f"without overlapping any scene object (clearance={clearance_m:.3f} m)."
                ), artifact

            lines = [
                f"COLLISION: '{object_name}' at ({x:.3f}, {y:.3f}, {z:.3f}) "
                f"would overlap {len(conflicts_list)} object(s):"
            ]
            for c in conflicts_list:
                lines.append(f"  • {c}")
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Segment 7 — Accessibility & Preconditions
# ---------------------------------------------------------------------------

class IsAccessibleInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object to check (e.g., 'Milk', 'milk.stl')."
    )

class IsAccessibleTool(AgenticTool):
    """Report whether an object is reachable by the gripper right now, naming any blocker."""

    name: str = "is_accessible"
    description: str = (
        "Check whether an object can be reached by the robot's end-effector right now — "
        "without first opening a container or moving a blocking object. "
        "Performs two checks in sequence: "
        "(1) containment — is the object inside a closed container (drawer, fridge, cabinet)? "
        "(2) stacking — is another object sitting on top of it, preventing a top-down grasp? "
        "Returns True/False and, when blocked, names the blocker and explains why "
        "(e.g. a closed container or an object stacked on top must be cleared first)."
    )
    args_schema: Type[BaseModel] = IsAccessibleInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Checking accessibility of: {object_name}")

            query = object_name.strip().lower()
            target_ann = _find_annotation_by_query(query, get_annotations())
            if target_ann is None:
                return f"Object '{object_name}' not found in semantic annotations.", None

            target_root = getattr(target_ann, "root", None)
            target_pos = _get_body_position(target_root) if target_root else None

            # Check 1: Containment + joint state
            for container_ann in get_annotations():
                if container_ann is target_ann:
                    continue
                if _is_robot_annotation(container_ann):
                    continue
                stored_objects = getattr(container_ann, "objects", None)
                if not stored_objects:
                    continue
                if target_ann not in stored_objects:
                    continue

                container_bodies = list(getattr(container_ann, "bodies", []))
                for body in container_bodies:
                    conn = getattr(body, "parent_connection", None)
                    if conn is None:
                        continue
                    if not (hasattr(conn, "position") and hasattr(conn, "is_controlled")):
                        continue
                    try:
                        pos_val = float(conn.position)
                        dof = getattr(conn, "dof", None)
                        lo = None
                        if dof is not None and hasattr(dof, "has_position_limits") and dof.has_position_limits():
                            lims = dof.limits
                            lo = float(lims.lower.position) if lims.lower is not None else None

                        if lo is not None and pos_val <= lo + 0.05:
                            name_obj = getattr(conn, "name", None)
                            jname = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
                            artifact = IsAccessibleArtifact(
                                object_name=object_name,
                                accessible=False,
                                blocker=container_ann.__class__.__name__,
                                reason="container_closed",
                                fix=f"open [{container_ann.__class__.__name__}] first",
                            )
                            return (
                                f"'{object_name}' is accessible: False\n"
                                f"Blocked by: [{container_ann.__class__.__name__}]  "
                                f"reason=container_closed  "
                                f"joint='{jname}'  position={pos_val:.3f}  "
                                f"(closed_limit={lo:.3f})\n"
                                f"Fix: open [{container_ann.__class__.__name__}] first."
                            ), artifact
                    except Exception:
                        continue

                artifact = IsAccessibleArtifact(object_name=object_name, accessible=True)
                return (
                    f"'{object_name}' is accessible: True\n"
                    f"Note: object is inside [{container_ann.__class__.__name__}], "
                    f"but its access point is currently open."
                ), artifact

            # Check 2: Stacking
            if target_pos is not None and target_root is not None:
                tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
                t_dims = symbol_bounding_box(target_root)
                t_hd = (t_dims[0] / 2.0) if t_dims else 0.1
                t_hw = (t_dims[1] / 2.0) if t_dims else 0.1
                t_hh = (t_dims[2] / 2.0) if t_dims else 0.1
                t_top_z = tz + t_hh

                xy_margin = 0.05

                for other_ann in get_annotations():
                    if other_ann is target_ann:
                        continue
                    if _is_robot_annotation(other_ann):
                        continue
                    other_root = getattr(other_ann, "root", None)
                    if other_root is None:
                        continue
                    other_pos = _get_body_position(other_root)
                    if other_pos is None:
                        continue
                    ox, oy, oz = float(other_pos[0]), float(other_pos[1]), float(other_pos[2])

                    within_x = abs(ox - tx) <= t_hd + xy_margin
                    within_y = abs(oy - ty) <= t_hw + xy_margin
                    above_top = t_top_z - 0.02 <= oz <= t_top_z + 0.35

                    if within_x and within_y and above_top:
                        other_bname = _extract_body_name(other_root)
                        artifact = IsAccessibleArtifact(
                            object_name=object_name,
                            accessible=False,
                            blocker=other_ann.__class__.__name__,
                            reason="object_on_top",
                            fix=f"move [{other_ann.__class__.__name__}] off the target first",
                        )
                        return (
                            f"'{object_name}' is accessible: False\n"
                            f"Blocked by: [{other_ann.__class__.__name__}]  "
                            f"body={other_bname}  "
                            f"reason=object_on_top  "
                            f"pos={_fmt_pos((ox, oy, oz))}\n"
                            f"Fix: move [{other_ann.__class__.__name__}] off the target first."
                        ), artifact

            artifact = IsAccessibleArtifact(object_name=object_name, accessible=True)
            return f"'{object_name}' is accessible: True", artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Segment 8 — Causal & Consequence Reasoning
# ---------------------------------------------------------------------------

class GetSupportingObjectInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object whose support to find "
                    "(e.g., 'Milk', 'milk.stl')."
    )

class GetSupportingObjectTool(AgenticTool):
    """Return what is physically supporting a given object (the surface or object it rests on)."""

    name: str = "get_supporting_object"
    description: str = (
        "Find the object or surface that is physically supporting (holding up) a given object. "
        "Returns the supporter annotation name, body name, and position. "
        "If nothing is found beneath the object it is assumed to be resting on the floor. "
        "Use this for causal reasoning: knowing what supports X tells you what must stay in "
        "place for X to remain stable, and what will be exposed if X is moved."
    )
    args_schema: Type[BaseModel] = GetSupportingObjectInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Finding supporting object for: {object_name}")

            query = object_name.strip().lower()
            target_ann = _find_annotation_by_query(query, get_annotations())
            if target_ann is None:
                return f"Object '{object_name}' not found in semantic annotations.", None

            target_root = getattr(target_ann, "root", None)
            target_pos = _get_body_position(target_root) if target_root else None
            if target_pos is None:
                return f"Could not retrieve world position for '{object_name}'.", None

            tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
            t_dims = symbol_bounding_box(target_root) if target_root else None
            t_hd = (t_dims[0] / 2.0) if t_dims else 0.1
            t_hw = (t_dims[1] / 2.0) if t_dims else 0.1
            t_hh = (t_dims[2] / 2.0) if t_dims else 0.1
            t_bottom_z = tz - t_hh

            z_tol = 0.06
            xy_margin = 0.06

            best_supporter = None
            best_supporter_pos = None

            for other_ann in get_annotations():
                if other_ann is target_ann:
                    continue
                if _is_robot_annotation(other_ann):
                    continue
                other_root = getattr(other_ann, "root", None)
                if other_root is None:
                    continue
                other_pos = _get_body_position(other_root)
                if other_pos is None:
                    continue
                ox, oy, oz = float(other_pos[0]), float(other_pos[1]), float(other_pos[2])

                other_dims = symbol_bounding_box(other_root)
                other_hd = (other_dims[0] / 2.0) if other_dims else 0.1
                other_hw = (other_dims[1] / 2.0) if other_dims else 0.1
                other_hh = (other_dims[2] / 2.0) if other_dims else 0.1
                other_top_z = oz + other_hh

                z_contact = abs(t_bottom_z - other_top_z) <= z_tol
                within_x = abs(tx - ox) <= other_hd + t_hd + xy_margin
                within_y = abs(ty - oy) <= other_hw + t_hw + xy_margin

                if z_contact and within_x and within_y:
                    if best_supporter is None or abs(t_bottom_z - other_top_z) < abs(
                        t_bottom_z - (best_supporter_pos[2] + 0.0)
                    ):
                        best_supporter = other_ann
                        best_supporter_pos = (ox, oy, oz)

            if best_supporter is not None:
                bname = _extract_body_name(getattr(best_supporter, "root", None))
                ox, oy, oz = best_supporter_pos
                artifact = SupportingObjectArtifact(
                    object_name=object_name,
                    supported_by=best_supporter.__class__.__name__,
                    supporter_body=bname,
                    supporter_position=[round(ox, 3), round(oy, 3), round(oz, 3)],
                )
                return (
                    f"'{object_name}' is supported by: [{best_supporter.__class__.__name__}]  "
                    f"body={bname}  pos={_fmt_pos((ox, oy, oz))}"
                ), artifact

            if t_bottom_z <= 0.08:
                artifact = SupportingObjectArtifact(
                    object_name=object_name,
                    supported_by="floor",
                    supporter_body=None,
                    supporter_position=None,
                )
                return f"'{object_name}' is resting on the floor/ground (bottom_z={t_bottom_z:.3f} m).", artifact

            artifact = SupportingObjectArtifact(
                object_name=object_name,
                supported_by=None,
                supporter_body=None,
                supporter_position=None,
            )
            return (
                f"'{object_name}' has no detected supporter "
                f"(bottom_z={t_bottom_z:.3f} m, no object found within contact tolerance)."
            ), artifact

        except Exception as e:
            return self._handle_error(e), None


# ---------------------------------------------------------------------------
# Tool: GetObjectsSupportedBy
# ---------------------------------------------------------------------------

class GetObjectsSupportedByInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object acting as a support "
                    "(e.g., 'CuttingBoard', 'cutting_board.stl', 'Tray')."
    )

class GetObjectsSupportedByTool(AgenticTool):
    """Return all objects that would be displaced if a given object were moved."""

    name: str = "get_objects_supported_by"
    description: str = (
        "Find all objects currently resting on top of a given object — even if that object "
        "is not annotated as a surface (e.g., a tray, a cutting board, a book). "
        "Complements get_objects_on_surface, which only works for surface-typed annotations. "
        "Use this before moving or picking up an object to identify what would be "
        "displaced or fall as a consequence."
    )
    args_schema: Type[BaseModel] = GetObjectsSupportedByInput

    def _run(self, object_name: str):
        try:
            logger.debug(f"[SDT Tool] Finding objects resting on: {object_name}")

            query = object_name.strip().lower()
            target_ann = _find_annotation_by_query(query, get_annotations())
            if target_ann is None:
                return f"Object '{object_name}' not found in semantic annotations.", None

            target_root = getattr(target_ann, "root", None)
            target_pos = _get_body_position(target_root) if target_root else None
            if target_pos is None:
                return f"Could not retrieve world position for '{object_name}'.", None

            tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
            t_dims = symbol_bounding_box(target_root) if target_root else None
            t_hd = (t_dims[0] / 2.0) if t_dims else 0.15
            t_hw = (t_dims[1] / 2.0) if t_dims else 0.15
            t_hh = (t_dims[2] / 2.0) if t_dims else 0.05
            t_top_z = tz + t_hh

            z_tol = 0.06
            xy_margin = 0.06

            resting = []

            for other_ann in get_annotations():
                if other_ann is target_ann:
                    continue
                if _is_robot_annotation(other_ann):
                    continue
                other_root = getattr(other_ann, "root", None)
                if other_root is None:
                    continue
                other_pos = _get_body_position(other_root)
                if other_pos is None:
                    continue
                ox, oy, oz = float(other_pos[0]), float(other_pos[1]), float(other_pos[2])

                other_dims = symbol_bounding_box(other_root)
                other_hh = (other_dims[2] / 2.0) if other_dims else 0.05
                other_bottom_z = oz - other_hh

                z_contact = abs(other_bottom_z - t_top_z) <= z_tol
                within_x = abs(ox - tx) <= t_hd + xy_margin
                within_y = abs(oy - ty) <= t_hw + xy_margin

                if z_contact and within_x and within_y:
                    bname = _extract_body_name(other_root)
                    resting.append((other_ann.__class__.__name__, bname, ox, oy, oz))

            if not resting:
                artifact = SupportedByArtifact(object_name=object_name, resting_objects=[])
                return f"No objects found resting on '{object_name}'.", artifact

            lines = [f"Objects supported by '{object_name}' ({len(resting)} found):"]
            resting_list = []
            for cls_name, bname, ox, oy, oz in resting:
                lines.append(f"  [{cls_name}]  body={bname}  pos={_fmt_pos((ox, oy, oz))}")
                resting_list.append({"cls_name": cls_name, "body_name": bname,
                                     "position": [round(ox, 3), round(oy, 3), round(oz, 3)]})
            lines.append(f"Caution: these objects would be displaced if '{object_name}' is moved.")

            artifact = SupportedByArtifact(object_name=object_name, resting_objects=resting_list)
            return "\n".join(lines), artifact

        except Exception as e:
            return self._handle_error(e), None
