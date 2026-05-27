"""Scene query tools — query the semantic scene graph for object poses, relations, and properties.
These tools use the `semantic_digital_twin` package natively.
"""

import functools
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool
from agentic_llmr.platform.world import (
    get_active_world, register_world_cache,
    _is_robot_annotation, sort_by_volume, get_annotations, get_bodies,
    symbol_display_name, get_arm_label, find_body_by_name, get_robot_base_body,
    symbol_bounding_box,
)

# ---------------------------------------------------------------------------
# Helpers for Semantic Digital Twin native objects
# ---------------------------------------------------------------------------

def _extract_body_name(body: Any) -> str:
    """Return the display name of a body, falling back to a unique id-based string."""
    return symbol_display_name(body) or f"body_{id(body)}"

def _extract_semantic_types(body: Any) -> List[str]:
    """Extract all semantic annotation class names for a Body via SymbolGraph."""
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
    """Return the (x, y, z) world-frame origin of a body via body.global_pose.

    No world parameter needed — the body carries its own _world back-reference.
    """
    try:
        pose = body.global_pose
        return float(pose.x), float(pose.y), float(pose.z)
    except Exception:
        return None

def _extract_body_pose(body: Any) -> Dict[str, Any]:
    """Extract the world-frame position and orientation of a body via body.global_pose.

    Uses body.global_pose (position: pose.x/y/z) and pose.to_quaternion() for orientation.
    No world parameter or scipy required.
    """
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
    """Return frozenset of all robot annotation class names via SymbolGraph."""
    names: set = set()
    for ann in get_annotations():
        if _is_robot_annotation(ann):
            for cls in type(ann).__mro__:
                if cls is object:
                    break
                names.add(cls.__name__)
    return frozenset(names)

register_world_cache(_build_robot_type_names.cache_clear)


# ---------------------------------------------------------------------------
# Segment 1 — World Inventory & Taxonomy
# ---------------------------------------------------------------------------

class SceneQueryInput(BaseModel):
    pass


class GetSceneObjectsTool(AgenticTool):
    name: str = "list_all_objects"
    description: str = (
        "Primary scene inventory primitive — returns every task-relevant object in the scene "
        "as a table of body_name, semantic type(s), and 3D position. "
        "Call this first for any scene overview before using more specific queries. "
        "Robot structural parts (arms, grippers, base links) are filtered out automatically. "
        "Use the returned body_name values as input to all other tools that require an exact object identifier."
    )
    args_schema: Type[BaseModel] = SceneQueryInput

    def _run(self) -> str:
        try:
            print("[SDT Tool] Querying native semantic scene graph...")
            _, robot_view = get_active_world()

            annotations = get_annotations()
            robot_type_names = _build_robot_type_names()

            # Collect body IDs that belong to at least one annotation
            annotated_ids = set()
            for ann in annotations:
                for b in getattr(ann, "bodies", []):
                    annotated_ids.add(id(b))

            lines = [
                "| body_name | semantic_types | xyz_position |",
                "| --- | --- | --- |"
            ]

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
                xyz = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "-"
                lines.append(f"| {name} | {types_str} | {xyz} |")
                found += 1

            robot_base = get_robot_base_body(robot_view)
            if robot_base is not None:
                rname = _extract_body_name(robot_base)
                rpos = _get_body_position(robot_base)
                rxyz = f"({rpos[0]:.3f}, {rpos[1]:.3f}, {rpos[2]:.3f})" if rpos else "-"
                lines.append(f"| {rname} | Robot | {rxyz} |")

            if found == 0:
                lines = ["| annotation | root_body | xyz_position |", "| --- | --- | --- |"]
                for ann in annotations:
                    root = getattr(ann, "root", None)
                    rname = _extract_body_name(root) if root else "-"
                    pos = _get_body_position(root) if root else None
                    xyz = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "-"
                    lines.append(f"| {ann.__class__.__name__} | {rname} | {xyz} |")

            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: GetSemanticAnnotations
# ---------------------------------------------------------------------------

class SemanticAnnotationsInput(BaseModel):
    pass


class GetSemanticAnnotationsTool(AgenticTool):
    name: str = "list_object_types"
    description: str = (
        "Type-catalog primitive — lists every semantic annotation class name currently active in the scene "
        "(e.g. Milk, Table, Cereal, Fridge) together with the body names associated with each type. "
        "Use this to discover valid type names when find_objects_by_type returns no match, "
        "or to get a quick inventory of what semantic categories are present. "
        "Does NOT return object positions — use list_all_objects or get_object_pose for that."
    )
    args_schema: Type[BaseModel] = SemanticAnnotationsInput

    def _run(self) -> str:
        try:
            print("[SDT Tool] Listing semantic annotations...")

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
                return "No semantic annotations found in the world."

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

            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: FindObjectsByType
# ---------------------------------------------------------------------------

class FindObjectsByTypeInput(BaseModel):
    semantic_type: str = Field(
        description="Semantic class name to search for (e.g., 'Table', 'Milk', 'Cup'). Case-insensitive partial match."
    )


class FindObjectsByTypeTool(AgenticTool):
    name: str = "find_objects_by_type"
    description: str = (
        "Resolves a semantic type name to all matching objects with their exact body names and world-frame positions. "
        "Case-insensitive partial match — 'milk' matches 'Milk', 'FreshMilk', etc. "
        "Always call this to convert a semantic name (e.g. 'Milk') into the body_name(s) required by all other tools. "
        "If no match is found, call list_object_types to discover the correct type name."
    )
    args_schema: Type[BaseModel] = FindObjectsByTypeInput

    def _run(self, semantic_type: str) -> str:
        try:
            print(f"[SDT Tool] Searching for semantic type: {semantic_type}")
            query = semantic_type.strip().lower()

            matches = []
            for ann in get_annotations():
                cls_name = ann.__class__.__name__
                if query not in cls_name.lower():
                    continue
                body_entries = []
                for body in getattr(ann, "bodies", []):
                    bname = _extract_body_name(body)
                    pos = _get_body_position(body)
                    pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "unknown"
                    body_entries.append(f"body_name='{bname}'  pos={pos_str}")
                matches.append((cls_name, body_entries))

            if not matches:
                return (
                    f"No objects matching type '{semantic_type}' found. "
                    "Use list_semantic_annotations to see all available types."
                )

            lines = [f"Found {len(matches)} annotation(s) matching '{semantic_type}':"]
            for cls_name, body_entries in matches:
                lines.append(f"  [{cls_name}]")
                for entry in body_entries:
                    lines.append(f"    {entry}")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: GetObjectType
# ---------------------------------------------------------------------------

class ObjectTypeInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object (e.g., 'milk.stl').")


class GetObjectTypeTool(AgenticTool):
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

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Getting semantic type for: {object_name}")

            target_body = find_body_by_name(object_name)
            if target_body is None:
                return f"Object '{object_name}' not found. Use list_all_objects to see valid body names."

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
                        matched.append((direct_type, " → ".join(hierarchy)))
                        break

            if not matched:
                return f"No semantic annotation found for '{object_name}'."

            lines = [f"Object '{object_name}':"]
            for direct_type, hierarchy_str in matched:
                lines.append(f"  Semantic type:   {direct_type}")
                lines.append(f"  Type hierarchy:  {hierarchy_str}")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: ClassifyObjectsByRole
# ---------------------------------------------------------------------------

class ClassifyByRoleInput(BaseModel):
    pass


class ClassifyObjectsByRoleTool(AgenticTool):
    name: str = "classify_objects_by_role"
    description: str = (
        "Scene-wide structural overview — categorises every object in the scene into one of four roles: "
        "surface (objects you can place things on, e.g. Table, Shelf), "
        "articulated (objects with controllable joints, e.g. Fridge, Drawer, Door), "
        "object (movable items, containers, tools — everything else), "
        "agent (the robot and its structural parts). "
        "Use this as a structural map of the scene before planning multi-step tasks. "
        "It tells you what can be opened, what can be placed on, and what can be picked up."
    )
    args_schema: Type[BaseModel] = ClassifyByRoleInput

    def _run(self) -> str:
        try:
            print("[SDT Tool] Classifying scene objects by role...")
            _, robot_view = get_active_world()

            surfaces = []
            articulateds = []
            objects = []
            agent_entries = []

            for ann in get_annotations():
                # Agent (robot structural parts) — highest priority
                if _is_robot_annotation(ann):
                    root = getattr(ann, "root", None)
                    rname = _extract_body_name(root) if root else "-"
                    agent_entries.append((ann.__class__.__name__, rname))
                    continue

                cls_name = ann.__class__.__name__
                body_names = [_extract_body_name(b) for b in getattr(ann, "bodies", [])]
                primary_name = body_names[0] if body_names else "-"

                # Compute position of root or first body
                pos = None
                root = getattr(ann, "root", None)
                ref_body = root if root is not None else (
                    getattr(ann, "bodies", [None])[0]
                )
                if ref_body is not None:
                    pos = _get_body_position(ref_body)
                pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "unknown"

                # Surface: has supporting_surface attribute
                if hasattr(ann, "supporting_surface"):
                    surfaces.append((cls_name, primary_name, pos_str))
                    continue

                # Articulated: any body has a controlled parent_connection
                is_articulated = False
                for b in getattr(ann, "bodies", []):
                    conn = getattr(b, "parent_connection", None)
                    if conn is not None and getattr(conn, "is_controlled", False):
                        is_articulated = True
                        break
                if is_articulated:
                    articulateds.append((cls_name, primary_name, pos_str))
                    continue

                # Everything else is a movable object
                objects.append((cls_name, primary_name, pos_str))

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
                return "No objects found in the current scene."

            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Segment 2 — Geometric & Spatial Properties
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: GetObjectPose
# ---------------------------------------------------------------------------

class ObjectPoseInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object (e.g., 'milk.stl').")

class GetObjectPoseTool(AgenticTool):
    name: str = "get_object_pose"
    description: str = "Get the 3D pose (position and orientation) of a specific object by its exact body_name."
    args_schema: Type[BaseModel] = ObjectPoseInput

    def _run(self, object_name: str) -> Dict[str, Any]:
        try:
            print(f"[SDT Tool] Getting pose for: {object_name}")
            target_body = find_body_by_name(object_name)
            if target_body is None:
                return {"error": f"Object '{object_name}' not found in the semantic digital twin."}
            return _extract_body_pose(target_body)
        except Exception as e:
            return self._handle_error(e)



# ---------------------------------------------------------------------------
# Tool: GetObjectDimensions
# ---------------------------------------------------------------------------

class ObjectDimensionsInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object (e.g., 'milk.stl').")

class GetObjectDimensionsTool(AgenticTool):
    name: str = "get_object_dimensions"
    description: str = (
        "Get the physical bounding box (width, depth, height in metres), centre of mass, "
        "and computed volume (width × depth × height in m³) of a specific object. "
        "Essential for grasp approach planning, placement decisions, and size comparisons."
    )
    args_schema: Type[BaseModel] = ObjectDimensionsInput

    def _run(self, object_name: str) -> Dict[str, Any]:
        try:
            print(f"[SDT Tool] Getting dimensions for: {object_name}")

            target_body = find_body_by_name(object_name)
            if target_body is None:
                return {"error": f"Object '{object_name}' not found."}

            result: Dict[str, Any] = {"body_name": object_name}

            # Bounding box — from combined collision mesh (body-local frame, no world needed).
            # combined_mesh.extents is a pure trimesh AABB computation with no FK.
            try:
                mesh = getattr(target_body, "combined_mesh", None)
                if mesh is not None:
                    ex = mesh.extents  # (x, y, z) extents in body-local frame
                    result["bounding_box_m"] = {
                        "width":  round(float(ex[0]), 4),
                        "depth":  round(float(ex[1]), 4),
                        "height": round(float(ex[2]), 4),
                    }
                    result["volume_m3"] = round(float(ex[0]) * float(ex[1]) * float(ex[2]), 6)
                else:
                    result["bounding_box_m"] = {"error": "No collision mesh available."}
            except Exception:
                result["bounding_box_m"] = {"error": "Could not compute bounding box."}

            # Centre of mass via body property (uses self._world internally — no explicit world param).
            try:
                com = target_body.center_of_mass
                result["center_of_mass"] = {
                    "x": round(float(com.x), 4),
                    "y": round(float(com.y), 4),
                    "z": round(float(com.z), 4),
                }
            except Exception:
                pos = _get_body_position(target_body)
                if pos:
                    result["center_of_mass"] = {
                        "x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4),
                        "note": "origin fallback (no mesh COM)",
                    }
                else:
                    result["center_of_mass"] = {"error": "Could not compute COM."}

            return result

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: GetObjectOrientation
# ---------------------------------------------------------------------------

class ObjectOrientationInput(BaseModel):
    object_name: str = Field(description="The exact body_name of the object to analyse.")

class GetObjectOrientationTool(AgenticTool):
    name: str = "get_object_orientation"
    description: str = (
        "Analyse the orientation of an object: whether it is upright, tilted, lying on its side, "
        "or inverted, plus the tilt angle from vertical. Critical for selecting the correct "
        "grasp approach direction."
    )
    args_schema: Type[BaseModel] = ObjectOrientationInput

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Analysing orientation of: {object_name}")

            body = find_body_by_name(object_name)
            if body is None:
                return f"Object '{object_name}' not found."

            T = np.array(body.global_transform.to_np())
            R = T[:3, :3]

            # Object's local Z-axis expressed in world frame
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

            # Roll-pitch-yaw from rotation matrix
            roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
            pitch = float(np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1))))
            yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

            return (
                f"Object '{object_name}' orientation:\n"
                f"  Status: {label}\n"
                f"  Tilt from vertical: {tilt_deg:.1f}°\n"
                f"  Local Z-axis in world: ({local_z_in_world[0]:.3f}, {local_z_in_world[1]:.3f}, {local_z_in_world[2]:.3f})\n"
                f"  RPY (deg): roll={roll:.1f}  pitch={pitch:.1f}  yaw={yaw:.1f}"
            )

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Shared helpers for new tools
# ---------------------------------------------------------------------------


def _rgb_to_color_name(r: float, g: float, b: float) -> str:
    """Approximate an RGB value as a human-readable color name."""
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
    name: str = "get_object_color"
    description: str = (
        "Get the visual color of an object from its geometry. "
        "Use this to disambiguate between multiple objects of the same type "
        "(e.g., 'the red cup' vs 'the blue cup')."
    )
    args_schema: Type[BaseModel] = ObjectColorInput

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Getting color of: {object_name}")

            body = find_body_by_name(object_name)
            if body is None:
                return f"Object '{object_name}' not found."

            shapes = getattr(body, "visual", None)
            if shapes is None:
                return f"'{object_name}' has no visual geometry."

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
                return f"No explicit color found on '{object_name}'. Color may be from a texture or mesh material."

            lines = [f"Visual color of '{object_name}':"]
            for c in colors_found:
                lines.append(f"  {c}")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)
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
    name: str = "get_spatial_relation"
    description: str = (
        "Compute the spatial relationship between two objects: direction (front/back/left/right/above/below) "
        "and exact distance. Use 'robot' as reference_name to get robot-centric directions. "
        "Essential for resolving instructions like 'the cup to the left of the bowl'."
    )
    args_schema: Type[BaseModel] = SpatialRelationInput

    def _run(self, object_name: str, reference_name: str) -> str:
        try:
            print(f"[SDT Tool] Computing spatial relation: '{object_name}' relative to '{reference_name}'")
            _, robot_view = get_active_world()

            obj_body = find_body_by_name(object_name)
            if obj_body is None:
                return f"Object '{object_name}' not found."

            use_robot = reference_name.strip().lower() in ("robot", "me", "self")
            if use_robot:
                ref_body = get_robot_base_body(robot_view)
                ref_label = "robot base"
            else:
                ref_body = find_body_by_name(reference_name)
                ref_label = reference_name

            if ref_body is None:
                return f"Reference '{reference_name}' not found."

            obj_pos_raw = _get_body_position(obj_body)
            ref_pos_raw = _get_body_position(ref_body)
            if obj_pos_raw is None:
                return f"Could not compute position for '{object_name}'."
            if ref_pos_raw is None:
                return f"Could not compute position for reference '{ref_label}'."
            obj_pos = np.array(obj_pos_raw)
            ref_pos = np.array(ref_pos_raw)
            delta = obj_pos - ref_pos  # vector from reference → object
            distance = float(np.linalg.norm(delta))

            # Try to express in robot/reference local frame for directional labels
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

            return (
                f"'{object_name}' is {direction_str} '{ref_label}' ({frame}).\n"
                f"Offset vector: dx={dx:.3f}m  dy={dy:.3f}m  dz={dz:.3f}m\n"
                f"Euclidean distance: {distance:.3f}m"
            )

        except Exception as e:
            return self._handle_error(e)

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

    def _run(self, object_name: str, max_count: int = 5, radius_m: Optional[float] = None) -> str:
        try:
            print(f"[SDT Tool] Finding nearest objects to: {object_name}")

            ref_body = find_body_by_name(object_name)
            if ref_body is None:
                return f"Reference object '{object_name}' not found."

            ref_pos = _get_body_position(ref_body)
            if ref_pos is None:
                return f"Could not compute position for '{object_name}'."
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
                return f"No other annotated objects found to compare with '{object_name}'."

            distances.sort(key=lambda x: x[0])
            if radius_m is not None:
                distances = [(d, c, n) for d, c, n in distances if d <= radius_m]
                if not distances:
                    return f"No objects found within {radius_m:.3f}m of '{object_name}'."

            lines = [f"Nearest objects to '{object_name}'" + (f" (within {radius_m:.3f}m)" if radius_m is not None else "") + ":"]
            for dist, cls_name, bname in distances[:max_count]:
                lines.append(f"  [{cls_name}]  body: {bname}  distance: {dist:.3f}m")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

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
    name: str = "get_objects_on_surface"
    description: str = (
        "Find all objects currently resting on a given surface (table, shelf, counter, etc.). "
        "Provide the semantic type name or exact body_name of the surface."
    )
    args_schema: Type[BaseModel] = SurfaceQueryInput

    def _run(self, surface_name: str) -> str:
        try:
            print(f"[SDT Tool] Querying objects on surface: {surface_name}")
            query = surface_name.strip().lower()

            annotations = get_annotations()
            surface_ann = None
            for ann in annotations:
                if query in ann.__class__.__name__.lower():
                    surface_ann = ann
                    break
            if surface_ann is None:
                for ann in annotations:
                    if any(_extract_body_name(b).lower() == query for b in getattr(ann, "bodies", [])):
                        surface_ann = ann
                        break
            if surface_ann is None:
                return f"No annotation named '{surface_name}' found in this scene."

            surface_root = getattr(surface_ann, "root", None)
            surface_pos = _get_body_position(surface_root) if surface_root else None
            if surface_pos is None:
                return f"Objects on {surface_ann.__class__.__name__}: none."

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
                return f"Objects on {surface_ann.__class__.__name__}: none."
            lines = [f"Objects on {surface_ann.__class__.__name__}:"]
            for ann, pos in found:
                rname = _extract_body_name(getattr(ann, "root", None))
                lines.append(
                    f"  [{type(ann).__name__}]  body={rname}  "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: SortObjectsBySize
# ---------------------------------------------------------------------------

class SortBySizeInput(BaseModel):
    semantic_type: str = Field(
        description="Semantic class name to filter by (e.g., 'Cup', 'Bottle'). Returns all matches sorted by volume."
    )
    largest_first: bool = Field(default=True, description="If true, largest object is listed first.")

class SortObjectsBySizeTool(AgenticTool):
    name: str = "sort_objects_by_size"
    description: str = (
        "Find all objects of a given semantic type and rank them by physical volume (largest to smallest). "
        "Use this to resolve instructions like 'the large cup' or 'the small bottle'."
    )
    args_schema: Type[BaseModel] = SortBySizeInput

    def _run(self, semantic_type: str, largest_first: bool = True) -> str:
        try:
            print(f"[SDT Tool] Sorting '{semantic_type}' objects by size...")

            query = semantic_type.strip().lower()
            candidates = [
                ann for ann in get_annotations()
                if query in ann.__class__.__name__.lower() and hasattr(ann, "root")
            ]

            if not candidates:
                return f"No '{semantic_type}' objects with root bodies found in the scene."

            sorted_anns = sort_by_volume(candidates, largest_first=largest_first)

            lines = [
                f"'{semantic_type}' objects sorted by size ({'largest' if largest_first else 'smallest'} first):"
            ]
            for i, ann in enumerate(sorted_anns, 1):
                body_names = [_extract_body_name(b) for b in getattr(ann, "bodies", [])]
                pos = None
                ref_body = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
                if ref_body is not None:
                    pos = _get_body_position(ref_body)
                pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "unknown"
                dims = symbol_bounding_box(ref_body) if ref_body is not None else None
                vol_str = f"  vol={dims[0]*dims[1]*dims[2]:.5f}m³" if dims else ""
                lines.append(f"  {i}. [{ann.__class__.__name__}]  body: {', '.join(body_names)}  pos: {pos_str}{vol_str}")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Segment 3 — Structural & Topological Relations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: GetArticulatedObjectJoints
# ---------------------------------------------------------------------------

class ArticulatedJointsInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the articulated object (e.g., 'Drawer', 'Fridge', 'Door')."
    )

class GetArticulatedObjectJointsTool(AgenticTool):
    name: str = "get_articulated_object_joints"
    description: str = (
        "Get the controllable joints of a non-robot articulated object (drawer, door, fridge, cabinet, etc.), "
        "including joint type, current position, and limits. "
        "Use this before planning an OpenAction or CloseAction. "
        "For robot arm joints, use get_joint_states instead."
    )
    args_schema: Type[BaseModel] = ArticulatedJointsInput

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Getting joints for articulated object: {object_name}")

            query = object_name.strip().lower()
            annotations = get_annotations()
            target_ann = None
            for ann in annotations:
                if query in ann.__class__.__name__.lower():
                    target_ann = ann
                    break
            if target_ann is None:
                for ann in annotations:
                    if any(_extract_body_name(b).lower() == query for b in getattr(ann, "bodies", [])):
                        target_ann = ann
                        break
            if target_ann is None:
                return f"No object '{object_name}' found in semantic annotations."

            ann_bodies = list(getattr(target_ann, "bodies", []))
            if not ann_bodies:
                return f"'{object_name}' annotation has no associated bodies."

            # Each body carries its own parent_connection — no world.connections needed.
            rows = []
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
                    if dof.has_position_limits():
                        lims = dof.limits
                        lo = lims.lower.position if lims.lower is not None else None
                        hi = lims.upper.position if lims.upper is not None else None
                        lim_str = f"[{float(lo):.3f}, {float(hi):.3f}]" if (lo is not None and hi is not None) else "limited"
                    else:
                        lim_str = "unlimited"
                    rows.append(f"  {jname}  type={jtype}  pos={pos}  limits={lim_str}")
                except Exception:
                    continue

            if not rows:
                return (
                    f"No active controllable joints found for '{target_ann.__class__.__name__}'. "
                    "It may be a rigid object."
                )

            lines = [f"Joints of [{target_ann.__class__.__name__}]:"]
            lines.extend(rows)
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

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

class GetContainedItemsTool(AgenticTool):
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

    def _run(self, container_name: str) -> str:
        try:
            print(f"[SDT Tool] Finding items contained in: {container_name}")
            query = container_name.strip().lower()

            annotations = get_annotations()

            # Find container annotation — semantic type match first, then body_name match
            container_ann = None
            for ann in annotations:
                if query in ann.__class__.__name__.lower():
                    container_ann = ann
                    break
            if container_ann is None:
                for ann in annotations:
                    if any(_extract_body_name(b).lower() == query
                           for b in getattr(ann, "bodies", [])):
                        container_ann = ann
                        break
            if container_ann is None:
                return f"No container named '{container_name}' found in the scene."

            # Resolve root body
            container_root = getattr(container_ann, "root", None)
            if container_root is None:
                bodies = getattr(container_ann, "bodies", [])
                container_root = bodies[0] if bodies else None
            if container_root is None:
                return f"Container '{container_ann.__class__.__name__}' has no root body."

            # FK transform of the container body
            T_c = np.array(container_root.global_transform.to_np())
            t_c = T_c[:3, 3]
            R_c = T_c[:3, :3]

            # Bounding box in container's own body frame
            dims = symbol_bounding_box(container_root)
            if dims is None:
                return (
                    f"Could not compute bounding box for '{container_ann.__class__.__name__}'. "
                    "Containment check unavailable."
                )
            depth, width, height = dims

            # ── Path 1: semantic storage list (HasStorageSpace.objects) ──────────
            # This is the authoritative list — objects explicitly registered in
            # the container via add_object(). Prefer it over geometry when populated.
            semantic_objects = getattr(container_ann, "objects", None)
            if semantic_objects:
                found = []
                for obj in semantic_objects:
                    if _is_robot_annotation(obj):
                        continue
                    ref = getattr(obj, "root", None)
                    if ref is None:
                        bodies = getattr(obj, "bodies", [])
                        ref = bodies[0] if bodies else None
                    if ref is None:
                        continue
                    pos = _get_body_position(ref)
                    found.append((obj.__class__.__name__, _extract_body_name(ref), pos))

                if found:
                    lines = [
                        f"Objects inside '{container_ann.__class__.__name__}' "
                        f"({len(found)} found, semantic list):"
                    ]
                    for cls_name, bname, pos in found:
                        pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "unknown"
                        lines.append(f"  [{cls_name}]  body={bname}  pos={pos_str}")
                    return "\n".join(lines)

            # ── Path 2: geometric containment fallback ────────────────────────
            # Used when the semantic list is absent or empty (objects placed
            # without explicit registration, or annotation lacks HasStorageSpace).
            container_body_ids = {id(b) for b in getattr(container_ann, "bodies", [])}

            # 5 cm XY margin; generous Z range handles both bottom-origin and
            # centre-origin body conventions without knowing which the model uses.
            xy_margin = 0.05
            z_lo = -(height / 2) - xy_margin
            z_hi = height + xy_margin

            found = []
            for ann in annotations:
                if ann is container_ann or _is_robot_annotation(ann):
                    continue

                ref = getattr(ann, "root", None)
                if ref is None:
                    bodies = getattr(ann, "bodies", [])
                    ref = bodies[0] if bodies else None
                if ref is None or id(ref) in container_body_ids:
                    continue

                pos_world = _get_body_position(ref)
                if pos_world is None:
                    continue

                p_local = R_c.T @ (np.array(pos_world) - t_c)

                if (abs(p_local[0]) <= depth / 2 + xy_margin and
                        abs(p_local[1]) <= width / 2 + xy_margin and
                        z_lo <= p_local[2] <= z_hi):
                    found.append((ann.__class__.__name__, _extract_body_name(ref), tuple(pos_world)))

            if not found:
                return f"No objects found inside '{container_ann.__class__.__name__}'."

            lines = [
                f"Objects inside '{container_ann.__class__.__name__}' "
                f"({len(found)} found, geometric detection):"
            ]
            for cls_name, bname, pos in found:
                lines.append(
                    f"  [{cls_name}]  body={bname}  "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Segment 4 — Functional State & Affordances
# (stubs — require SDT to model dynamic object state before implementation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: GetObjectState  [STUB]
# Queries dynamic functional state attributes of an object annotation:
# temperature, fill_level, is_clean, is_powered, is_on, etc.
# Blocked on: SDT annotations do not currently model dynamic state.
# When SDT adds these attributes, this tool reads them via hasattr inspection
# on the annotation instance and returns all non-structural attributes.
# ---------------------------------------------------------------------------

class ObjectStateInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object (e.g., 'Milk', 'milk.stl')."
    )

class GetObjectStateTool(AgenticTool):
    name: str = "get_object_state"
    description: str = (
        "Returns the dynamic functional state of an object: temperature, fill level, "
        "cleanliness, power state, etc. "
        "[NOT YET IMPLEMENTED — requires SDT dynamic state modeling]"
    )
    args_schema: Type[BaseModel] = ObjectStateInput

    def _run(self, object_name: str) -> str:
        return (
            "get_object_state is not yet implemented. "
            "The SDT framework does not currently model dynamic object state "
            "(temperature, fill_level, is_clean, is_powered, etc.). "
            "This tool will be completed once those attributes are added to the annotation classes."
        )


# ---------------------------------------------------------------------------
# Tool: GetObjectAffordances  [STUB]
# Returns a structured affordance profile for an object: what actions can be
# performed on it (pick up, open, fill, place on, stack on, etc.).
# Partial implementation possible via type-hierarchy inspection (HasDoors,
# HasStorageSpace, HasSupportingSurface, HasHandle in MRO). Full implementation
# requires SDT to expose explicit affordance attributes and reachability data
# from Segment 7 (is_pickable depends on kinematic accessibility).
# ---------------------------------------------------------------------------

class ObjectAffordancesInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object (e.g., 'Fridge', 'cup.stl')."
    )

class GetObjectAffordancesTool(AgenticTool):
    name: str = "get_object_affordances"
    description: str = (
        "Returns a structured affordance profile for an object: whether it can be picked up, "
        "opened, filled, placed on, or used as a container. "
        "[PARTIAL STUB — type-hierarchy affordances derivable now; "
        "is_pickable requires Segment 7 reachability]"
    )
    args_schema: Type[BaseModel] = ObjectAffordancesInput

    def _run(self, object_name: str) -> str:
        return (
            "get_object_affordances is not yet fully implemented. "
            "In the meantime, derive affordances from: "
            "(1) get_object_type — check MRO for HasSupportingSurface, HasStorageSpace, "
            "HasDoors, HasDrawers, HasHandle, HasHinge, HasSlider; "
            "(2) get_articulated_object_joints — confirms openable joints; "
            "(3) classify_objects_by_role — confirms surface/articulated/object role."
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Segment 5 — Robot & Interaction State
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: GetJointStates
# ---------------------------------------------------------------------------

class JointStatesInput(BaseModel):
    pass

class GetJointStatesTool(AgenticTool):
    name: str = "get_joint_states"
    description: str = (
        "Get the current position and limits of all active robot arm joints (shoulder, elbow, wrist, etc.). "
        "Returns only the robot's own controllable arm joints — not joints of furniture or objects. "
        "Use this to inspect the robot's current arm configuration (e.g. parked vs extended). "
        "For joints of non-robot articulated objects like drawers or doors, "
        "use get_articulated_object_joints instead."
    )
    args_schema: Type[BaseModel] = JointStatesInput

    def _run(self) -> str:
        try:
            print("[SDT Tool] Reading active joint states...")
            _, robot_view = get_active_world()

            # Traverse robot view parts directly to avoid world.connections which can
            # hang if the world's kinematic graph is in an inconsistent state.
            rows = []
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
                        # Duck-typed ActiveConnection1DOF check
                        if not (hasattr(conn, "position") and hasattr(conn, "dof") and hasattr(conn, "is_controlled")):
                            continue
                        if not getattr(conn, "is_controlled", False):
                            continue
                        try:
                            name_obj = getattr(conn, "name", None)
                            name = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
                            pos = round(float(conn.position), 5)
                            dof = conn.dof
                            if dof.has_position_limits():
                                lims = dof.limits
                                lo = lims.lower.position if lims.lower is not None else None
                                hi = lims.upper.position if lims.upper is not None else None
                                lim_str = f"[{float(lo):.3f}, {float(hi):.3f}]" if (lo is not None and hi is not None) else "limited"
                            else:
                                lim_str = "unlimited"
                            rows.append(f"  {name}: position={pos}  limits={lim_str}")
                        except Exception:
                            continue

            if not rows:
                return "No controllable joints found via robot model traversal."

            return "Active Joint States:\n" + "\n".join(rows)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: GetRobotPose
# ---------------------------------------------------------------------------

class RobotPoseInput(BaseModel):
    pass

class GetRobotPoseTool(AgenticTool):
    name: str = "get_robot_pose"
    description: str = (
        "Get the current world-frame position and orientation of the robot base. "
        "Use this as a reference for distance calculations or spatial reasoning "
        "relative to the robot. Returns position (x, y, z) and orientation as quaternion."
    )
    args_schema: Type[BaseModel] = RobotPoseInput

    def _run(self) -> str:
        try:
            print("[SDT Tool] Getting robot base pose...")
            _, robot_view = get_active_world()
            from scipy.spatial.transform import Rotation as R

            base_body = get_robot_base_body(robot_view)
            if base_body is None:
                return "Could not locate robot base body."

            T = np.array(base_body.global_transform.to_np())
            pos = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
            bname = _extract_body_name(base_body)
            return (
                f"Robot base ({bname}):\n"
                f"  position: ({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f})\n"
                f"  orientation (xyzw): ({float(quat[0]):.3f}, {float(quat[1]):.3f}, "
                f"{float(quat[2]):.3f}, {float(quat[3]):.3f})"
            )

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: GetEndEffectorPose
# ---------------------------------------------------------------------------

class EndEffectorPoseInput(BaseModel):
    arm: str = Field(
        description="Which arm to query: 'left' or 'right'."
    )

class GetEndEffectorPoseTool(AgenticTool):
    name: str = "get_end_effector_pose"
    description: str = (
        "Get the current world-frame pose of the robot's gripper tool frame for a given arm. "
        "Use this to know where the gripper is right now — useful before grasping, "
        "for proximity checks, or to understand the current arm configuration geometrically."
    )
    args_schema: Type[BaseModel] = EndEffectorPoseInput

    def _run(self, arm: str) -> str:
        try:
            print(f"[SDT Tool] Getting end-effector pose for: {arm} arm")
            _, robot_view = get_active_world()
            from scipy.spatial.transform import Rotation as R

            side = arm.strip().lower()
            target_arm = None
            for a in getattr(robot_view, "arms", []):
                if get_arm_label(a, robot_view) == side:
                    target_arm = a
                    break

            if target_arm is None:
                return f"No arm matching '{arm}' found. Use 'left' or 'right'."

            manip = getattr(target_arm, "manipulator", None)
            tool_frame = getattr(manip, "tool_frame", None) if manip else None
            if tool_frame is None:
                tool_frame = getattr(target_arm, "tip", None)
            if tool_frame is None:
                return f"Could not find tool frame for {arm} arm."

            T = np.array(tool_frame.global_transform.to_np())
            pos = T[:3, 3]
            quat = R.from_matrix(T[:3, :3]).as_quat()  # x, y, z, w
            tf_name = _extract_body_name(tool_frame)
            return (
                f"{side.upper()} end-effector ({tf_name}):\n"
                f"  position: ({float(pos[0]):.3f}, {float(pos[1]):.3f}, {float(pos[2]):.3f})\n"
                f"  orientation (xyzw): ({float(quat[0]):.3f}, {float(quat[1]):.3f}, "
                f"{float(quat[2]):.3f}, {float(quat[3]):.3f})"
            )

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
# Tool: GetGripperState
# ---------------------------------------------------------------------------

class GripperStateInput(BaseModel):
    arm: str = Field(
        description="Which arm's gripper to query: 'left' or 'right'."
    )

class GetGripperStateTool(AgenticTool):
    name: str = "get_gripper_state"
    description: str = (
        "Get the current opening state of the robot's gripper for a given arm. "
        "Returns whether the gripper is open, closed, or partially open, "
        "and the approximate opening width in metres. "
        "To determine what object is being held, use get_held_object instead."
    )
    args_schema: Type[BaseModel] = GripperStateInput

    def _run(self, arm: str) -> str:
        try:
            print(f"[SDT Tool] Getting gripper state for: {arm} arm")
            _, robot_view = get_active_world()

            side = arm.strip().lower()
            target_arm = None
            for a in getattr(robot_view, "arms", []):
                if get_arm_label(a, robot_view) == side:
                    target_arm = a
                    break

            if target_arm is None:
                return f"No arm matching '{arm}' found. Use 'left' or 'right'."

            manip = getattr(target_arm, "manipulator", None)
            if manip is None:
                return f"No gripper/manipulator attached to {arm} arm."

            gripper_type = type(manip).__name__

            # Physical opening width: FK distance between finger tip and thumb tip.
            # Duck-typed: ParallelGripper has finger+thumb, HumanoidGripper has fingers list+thumb.
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

            # Joint range percentage: first controlled connection in any gripper joint state.
            pct = None
            seen: set = set()
            for js in getattr(manip, "joint_states", []):
                for conn in getattr(js, "connections", []):
                    if id(conn) in seen:
                        continue
                    seen.add(id(conn))
                    # Duck-typed ActiveConnection1DOF check
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
                return f"Could not determine gripper state for {side} arm ({gripper_type})."

            if pct is not None:
                status = "closed" if pct < 5 else ("fully open" if pct > 90 else "partially open")
            else:
                status = "closed" if width_m < 0.005 else ("fully open" if width_m > 0.07 else "partially open")

            parts = [f"{side.upper()} gripper ({gripper_type}): {status}"]
            if width_m is not None:
                parts.append(f"  finger-to-thumb gap: {width_m:.4f}m")
            if pct is not None:
                parts.append(f"  joint range used: {pct:.1f}%")
            return "\n".join(parts)

        except Exception as e:
            return self._handle_error(e)


# ---------------------------------------------------------------------------
# Tool: GetHeldObject
# ---------------------------------------------------------------------------

class HeldObjectInput(BaseModel):
    arm: str = Field(description="Which arm to query: 'left' or 'right'.")

class GetHeldObjectTool(AgenticTool):
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

    def _run(self, arm: str) -> str:
        try:
            print(f"[SDT Tool] Checking held object for: {arm} arm")
            _, robot_view = get_active_world()
            side = arm.strip().lower()

            # Locate the target arm
            target_arm = None
            for a in getattr(robot_view, "arms", []):
                if get_arm_label(a, robot_view) == side:
                    target_arm = a
                    break
            if target_arm is None:
                return f"No arm matching '{arm}' found. Use 'left' or 'right'."

            # Collect every body ID that belongs to this arm's gripper structure.
            # When grasping, PyCRAM re-parents the object to one of these bodies.
            gripper_ids = _collect_gripper_body_ids(target_arm)
            if not gripper_ids:
                return f"Could not resolve gripper bodies for {side} arm."

            # Scan all non-robot annotations for one whose root body has a
            # FixedConnection parent inside the gripper.
            for ann in get_annotations():
                if _is_robot_annotation(ann):
                    continue
                root = getattr(ann, "root", None)
                if root is None:
                    continue
                conn = getattr(root, "parent_connection", None)
                if conn is None:
                    continue
                # Active connections (joints) have is_controlled; skip them.
                # Fixed attachment from grasping does NOT have is_controlled.
                if hasattr(conn, "is_controlled"):
                    continue
                parent_body = getattr(conn, "parent", None)
                if parent_body is not None and id(parent_body) in gripper_ids:
                    bname = _extract_body_name(root)
                    pos = _get_body_position(root)
                    pos_str = (
                        f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos else "unknown"
                    )
                    return (
                        f"{side.upper()} arm is holding: [{ann.__class__.__name__}]  "
                        f"body={bname}  pos={pos_str}"
                    )

            return f"{side.upper()} arm is not holding any object."

        except Exception as e:
            return self._handle_error(e)


def _collect_gripper_body_ids(arm: Any) -> set:
    """Return the set of body IDs that form the gripper/end-effector of the given arm.

    Uses only duck-typed attribute access — no SDT imports required.
    Covers ParallelGripper (finger + thumb) and HumanoidGripper (fingers list + thumb).
    """
    ids: set = set()

    # Arm-level tip (some robots expose this directly)
    tip = getattr(arm, "tip", None)
    if tip is not None:
        ids.add(id(tip))

    manip = getattr(arm, "manipulator", None)
    if manip is None:
        return ids

    # Tool frame — the primary kinematic attachment point for grasping
    tool_frame = getattr(manip, "tool_frame", None)
    if tool_frame is not None:
        ids.add(id(tool_frame))

    # ParallelGripper: single finger + single thumb
    for part_attr in ("finger", "thumb"):
        part = getattr(manip, part_attr, None)
        if part is None:
            continue
        for body_attr in ("root", "tip"):
            b = getattr(part, body_attr, None)
            if b is not None:
                ids.add(id(b))

    # HumanoidGripper: fingers list + thumb
    for f in getattr(manip, "fingers", []):
        for body_attr in ("root", "tip"):
            b = getattr(f, body_attr, None)
            if b is not None:
                ids.add(id(b))

    # Sweep all bodies reachable via manipulator joint_states connections
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

# ---------------------------------------------------------------------------
# Tool: CheckSceneCollisions
# ---------------------------------------------------------------------------

class SceneCollisionsInput(BaseModel):
    pass

class CheckSceneCollisionsTool(AgenticTool):
    name: str = "check_scene_collisions"
    description: str = (
        "Check whether any objects in the current scene are in collision. "
        "Returns all colliding body pairs and their penetration distances. "
        "Use this to verify a configuration is collision-free before executing."
    )
    args_schema: Type[BaseModel] = SceneCollisionsInput

    def _run(self) -> str:
        try:
            print("[SDT Tool] Checking scene collisions...")
            world, _ = get_active_world()

            result = world.collision_manager.compute_collisions()

            if not result.any():
                return "No collisions detected in the current scene."

            lines = [f"Collisions detected ({len(result.contacts)} contact(s)):"]
            for contact in result.contacts:
                name_a = _extract_body_name(contact.body_a)
                name_b = _extract_body_name(contact.body_b)
                dist = round(float(contact.distance), 5)
                status = "PENETRATING" if dist < 0 else "touching"
                lines.append(f"  {name_a}  <-->  {name_b}  distance={dist}m  [{status}]")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

# ---------------------------------------------------------------------------
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

class GetFreePlacementSpotsTool(AgenticTool):
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
    ) -> str:
        try:
            print(f"[SDT Tool] Sampling placement spots on: {surface_name}")

            query = surface_name.strip().lower()
            surface_ann = None
            for ann in get_annotations():
                if hasattr(ann, "supporting_surface") and query in ann.__class__.__name__.lower():
                    surface_ann = ann
                    break
            if surface_ann is None:
                for ann in get_annotations():
                    if hasattr(ann, "supporting_surface"):
                        if any(_extract_body_name(b).lower() == query for b in getattr(ann, "bodies", [])):
                            surface_ann = ann
                            break
            if surface_ann is None:
                return f"No annotation named '{surface_name}' with placement support exists in this scene."

            # Generate placement candidates purely from the surface root body's geometry.
            # Avoids sample_points_from_surface which internally calls world.modify_world()
            # and can corrupt the FK cache.
            root_body = getattr(surface_ann, "root", None)
            if root_body is None:
                return f"Surface '{surface_name}' has no root body."

            surface_pose = root_body.global_pose
            cx, cy, cz = float(surface_pose.x), float(surface_pose.y), float(surface_pose.z)

            mesh = getattr(root_body, "combined_mesh", None)
            if mesh is not None:
                ex, ey, ez = [float(v) for v in mesh.extents]
                half_x = ex * 0.4
                half_y = ey * 0.4
                z_top = cz + ez * 0.5
            else:
                half_x, half_y, z_top = 0.3, 0.3, cz + 0.02

            # Build a dense grid and select the first `count` unoccupied spots
            grid_n = max(4, int((count * 4) ** 0.5) + 1)
            xs = np.linspace(cx - half_x, cx + half_x, grid_n)
            ys = np.linspace(cy - half_y, cy + half_y, grid_n)
            candidates = [(float(x), float(y), z_top) for x in xs for y in ys]

            if check_occupancy:
                # Collect (px, py, half_w, half_d) for every object on the surface.
                # Uses the same on-surface heuristic as GetObjectsOnSurface.
                occupied: list[tuple[float, float, float, float]] = []
                for ann in get_annotations():
                    if _is_robot_annotation(ann):
                        continue
                    if ann is surface_ann:
                        continue
                    ref = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
                    if ref is None:
                        continue
                    pos = _get_body_position(ref)
                    if pos is None:
                        continue
                    # Only consider objects at roughly the same height as the surface top
                    if abs(pos[2] - z_top) > 0.25:
                        continue
                    dims = symbol_bounding_box(ref)
                    hw = (dims[1] / 2.0) if dims else 0.1
                    hd = (dims[0] / 2.0) if dims else 0.1
                    occupied.append((float(pos[0]), float(pos[1]), hw, hd))

                clearance = object_footprint_m
                free_candidates = []
                for cx_c, cy_c, cz_c in candidates:
                    blocked = False
                    for ox, oy, hw, hd in occupied:
                        if (abs(cx_c - ox) < hw + clearance and
                                abs(cy_c - oy) < hd + clearance):
                            blocked = True
                            break
                    if not blocked:
                        free_candidates.append((cx_c, cy_c, cz_c))
                candidates = free_candidates

            selected = candidates[:count]
            if not selected:
                return (
                    f"No free placement spots found on [{surface_ann.__class__.__name__}] "
                    f"with footprint clearance {object_footprint_m:.2f} m. "
                    "Try a smaller object_footprint_m or check_occupancy=False."
                )

            filtered_note = " (occupancy-filtered)" if check_occupancy else " (raw grid)"
            lines = [
                f"Candidate placement positions on [{surface_ann.__class__.__name__}]{filtered_note}:"
            ]
            for i, (x, y, z) in enumerate(selected, 1):
                lines.append(f"  {i}. position: ({x:.3f}, {y:.3f}, {z:.3f})")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

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
    ) -> str:
        try:
            print(f"[SDT Tool] Checking hypothetical placement of '{object_name}' at ({x:.3f},{y:.3f},{z:.3f})")

            # Resolve the object being placed
            obj_body = find_body_by_name(object_name)
            if obj_body is None:
                # Try annotation class-name match
                for ann in get_annotations():
                    if object_name.strip().lower() in ann.__class__.__name__.lower():
                        obj_body = getattr(ann, "root", None) or (getattr(ann, "bodies", [None])[0])
                        break
            if obj_body is None:
                return f"Object '{object_name}' not found in the scene."

            # Get the object's half-extents at the target pose
            dims = symbol_bounding_box(obj_body)
            if dims:
                obj_hd = dims[0] / 2.0 + clearance_m   # depth half-extent
                obj_hw = dims[1] / 2.0 + clearance_m   # width half-extent
                obj_hh = dims[2] / 2.0 + clearance_m   # height half-extent
            else:
                # Fall back to collision geometry extents
                mesh = getattr(obj_body, "combined_mesh", None)
                if mesh is not None:
                    ex, ey, ez = [float(v) / 2.0 + clearance_m for v in mesh.extents]
                    obj_hd, obj_hw, obj_hh = ex, ey, ez
                else:
                    obj_hd = obj_hw = obj_hh = 0.1 + clearance_m

            # Find the annotation that owns obj_body so we can skip it in the loop
            obj_ann_id = None
            for ann in get_annotations():
                bodies = getattr(ann, "bodies", [])
                root = getattr(ann, "root", None)
                if root is obj_body or obj_body in bodies:
                    obj_ann_id = id(ann)
                    break

            conflicts: list[str] = []

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

                # Get other object's half-extents (world-frame AABB approximation)
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

                # AABB overlap test in world frame
                ox, oy, oz = float(other_pos[0]), float(other_pos[1]), float(other_pos[2])
                overlap_x = abs(x - ox) < (obj_hd + other_hd)
                overlap_y = abs(y - oy) < (obj_hw + other_hw)
                overlap_z = abs(z - oz) < (obj_hh + other_hh)

                if overlap_x and overlap_y and overlap_z:
                    bname = _extract_body_name(ref)
                    conflicts.append(f"[{ann.__class__.__name__}] body={bname}")

            if not conflicts:
                return (
                    f"No collision: '{object_name}' can be placed at ({x:.3f}, {y:.3f}, {z:.3f}) "
                    f"without overlapping any scene object (clearance={clearance_m:.3f} m)."
                )

            lines = [
                f"COLLISION: '{object_name}' at ({x:.3f}, {y:.3f}, {z:.3f}) "
                f"would overlap {len(conflicts)} object(s):"
            ]
            for c in conflicts:
                lines.append(f"  • {c}")
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)


# ---------------------------------------------------------------------------
# Segment 7 — Accessibility & Preconditions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: IsAccessible
# ---------------------------------------------------------------------------

class IsAccessibleInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object to check (e.g., 'Milk', 'milk.stl')."
    )

class IsAccessibleTool(AgenticTool):
    name: str = "is_accessible"
    description: str = (
        "Check whether an object can be reached by the robot's end-effector right now — "
        "without first opening a container or moving a blocking object. "
        "Performs two checks in sequence: "
        "(1) containment — is the object inside a closed container (drawer, fridge, cabinet)? "
        "(2) stacking — is another object sitting on top of it, preventing a top-down grasp? "
        "Returns True/False and, when blocked, names the blocker and explains why. "
        "Use this before planning a PickUpAction to determine whether precondition steps "
        "(OpenAction, PlaceAction) are needed first."
    )
    args_schema: Type[BaseModel] = IsAccessibleInput

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Checking accessibility of: {object_name}")

            # ── Resolve target annotation ────────────────────────────────────
            query = object_name.strip().lower()
            target_ann = None
            for ann in get_annotations():
                if query in ann.__class__.__name__.lower():
                    target_ann = ann
                    break
            if target_ann is None:
                for ann in get_annotations():
                    if any(_extract_body_name(b).lower() == query
                           for b in getattr(ann, "bodies", [])):
                        target_ann = ann
                        break
            if target_ann is None:
                return f"Object '{object_name}' not found in semantic annotations."

            target_root = getattr(target_ann, "root", None)
            target_pos = _get_body_position(target_root) if target_root else None

            # ── Check 1: Containment + joint state ──────────────────────────
            # Annotations with HasStorageSpace expose .objects (semantic stored-items list).
            # If target_ann is registered there, check whether the container's joints are closed.
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

                # Target is inside this container — check if any controllable joint is closed
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

                        # Joint is "closed" when at or very near its lower limit
                        if lo is not None and pos_val <= lo + 0.05:
                            name_obj = getattr(conn, "name", None)
                            jname = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
                            return (
                                f"'{object_name}' is accessible: False\n"
                                f"Blocked by: [{container_ann.__class__.__name__}]  "
                                f"reason=container_closed  "
                                f"joint='{jname}'  position={pos_val:.3f}  "
                                f"(closed_limit={lo:.3f})\n"
                                f"Fix: open [{container_ann.__class__.__name__}] first."
                            )
                    except Exception:
                        continue

                # Container found but all joints are open — still reachable
                return (
                    f"'{object_name}' is accessible: True\n"
                    f"Note: object is inside [{container_ann.__class__.__name__}], "
                    f"but its access point is currently open."
                )

            # ── Check 2: Stacking — is something resting on top of the target? ─
            if target_pos is not None and target_root is not None:
                tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
                t_dims = symbol_bounding_box(target_root)   # (depth, width, height)
                t_hd = (t_dims[0] / 2.0) if t_dims else 0.1
                t_hw = (t_dims[1] / 2.0) if t_dims else 0.1
                t_hh = (t_dims[2] / 2.0) if t_dims else 0.1
                t_top_z = tz + t_hh

                xy_margin = 0.05  # 5 cm lateral tolerance

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
                        return (
                            f"'{object_name}' is accessible: False\n"
                            f"Blocked by: [{other_ann.__class__.__name__}]  "
                            f"body={other_bname}  "
                            f"reason=object_on_top  "
                            f"pos=({ox:.3f}, {oy:.3f}, {oz:.3f})\n"
                            f"Fix: move [{other_ann.__class__.__name__}] off the target first."
                        )

            return f"'{object_name}' is accessible: True"

        except Exception as e:
            return self._handle_error(e)


# ---------------------------------------------------------------------------
# Segment 8 — Causal & Consequence Reasoning
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tool: GetSupportingObject
# ---------------------------------------------------------------------------

class GetSupportingObjectInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object whose support to find "
                    "(e.g., 'Milk', 'milk.stl')."
    )

class GetSupportingObjectTool(AgenticTool):
    name: str = "get_supporting_object"
    description: str = (
        "Find the object or surface that is physically supporting (holding up) a given object. "
        "Returns the supporter annotation name, body name, and position. "
        "If nothing is found beneath the object it is assumed to be resting on the floor. "
        "Use this for causal reasoning: knowing what supports X tells you what must stay in "
        "place for X to remain stable, and what will be exposed if X is moved."
    )
    args_schema: Type[BaseModel] = GetSupportingObjectInput

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Finding supporting object for: {object_name}")

            # ── Resolve target ───────────────────────────────────────────────
            query = object_name.strip().lower()
            target_ann = None
            for ann in get_annotations():
                if query in ann.__class__.__name__.lower():
                    target_ann = ann
                    break
            if target_ann is None:
                for ann in get_annotations():
                    if any(_extract_body_name(b).lower() == query
                           for b in getattr(ann, "bodies", [])):
                        target_ann = ann
                        break
            if target_ann is None:
                return f"Object '{object_name}' not found in semantic annotations."

            target_root = getattr(target_ann, "root", None)
            target_pos = _get_body_position(target_root) if target_root else None
            if target_pos is None:
                return f"Could not retrieve world position for '{object_name}'."

            tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
            t_dims = symbol_bounding_box(target_root) if target_root else None
            t_hd = (t_dims[0] / 2.0) if t_dims else 0.1   # depth half-extent
            t_hw = (t_dims[1] / 2.0) if t_dims else 0.1   # width half-extent
            t_hh = (t_dims[2] / 2.0) if t_dims else 0.1   # height half-extent
            t_bottom_z = tz - t_hh                          # z of target's bottom face

            z_tol = 0.06    # ± 6 cm contact tolerance
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

                # Contact condition: target's bottom face rests on other's top face
                z_contact = abs(t_bottom_z - other_top_z) <= z_tol
                within_x = abs(tx - ox) <= other_hd + t_hd + xy_margin
                within_y = abs(ty - oy) <= other_hw + t_hw + xy_margin

                if z_contact and within_x and within_y:
                    # Prefer the object whose top face is closest to target's bottom
                    if best_supporter is None or abs(t_bottom_z - other_top_z) < abs(
                        t_bottom_z - (best_supporter_pos[2] + 0.0)
                    ):
                        best_supporter = other_ann
                        best_supporter_pos = (ox, oy, oz)

            if best_supporter is not None:
                bname = _extract_body_name(getattr(best_supporter, "root", None))
                ox, oy, oz = best_supporter_pos
                return (
                    f"'{object_name}' is supported by: [{best_supporter.__class__.__name__}]  "
                    f"body={bname}  pos=({ox:.3f}, {oy:.3f}, {oz:.3f})"
                )

            # No supporter found — check whether the object is near floor level
            if t_bottom_z <= 0.08:
                return f"'{object_name}' is resting on the floor/ground (bottom_z={t_bottom_z:.3f} m)."

            return (
                f"'{object_name}' has no detected supporter "
                f"(bottom_z={t_bottom_z:.3f} m, no object found within contact tolerance)."
            )

        except Exception as e:
            return self._handle_error(e)


# ---------------------------------------------------------------------------
# Tool: GetObjectsSupportedBy
# ---------------------------------------------------------------------------

class GetObjectsSupportedByInput(BaseModel):
    object_name: str = Field(
        description="Semantic type name or body_name of the object acting as a support "
                    "(e.g., 'CuttingBoard', 'cutting_board.stl', 'Tray')."
    )

class GetObjectsSupportedByTool(AgenticTool):
    name: str = "get_objects_supported_by"
    description: str = (
        "Find all objects currently resting on top of a given object — even if that object "
        "is not annotated as a surface (e.g., a tray, a cutting board, a book). "
        "Complements get_objects_on_surface, which only works for surface-typed annotations. "
        "Use this before moving or picking up an object to identify what would be "
        "displaced or fall as a consequence."
    )
    args_schema: Type[BaseModel] = GetObjectsSupportedByInput

    def _run(self, object_name: str) -> str:
        try:
            print(f"[SDT Tool] Finding objects resting on: {object_name}")

            # ── Resolve target ───────────────────────────────────────────────
            query = object_name.strip().lower()
            target_ann = None
            for ann in get_annotations():
                if query in ann.__class__.__name__.lower():
                    target_ann = ann
                    break
            if target_ann is None:
                for ann in get_annotations():
                    if any(_extract_body_name(b).lower() == query
                           for b in getattr(ann, "bodies", [])):
                        target_ann = ann
                        break
            if target_ann is None:
                return f"Object '{object_name}' not found in semantic annotations."

            target_root = getattr(target_ann, "root", None)
            target_pos = _get_body_position(target_root) if target_root else None
            if target_pos is None:
                return f"Could not retrieve world position for '{object_name}'."

            tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
            t_dims = symbol_bounding_box(target_root) if target_root else None
            t_hd = (t_dims[0] / 2.0) if t_dims else 0.15
            t_hw = (t_dims[1] / 2.0) if t_dims else 0.15
            t_hh = (t_dims[2] / 2.0) if t_dims else 0.05
            t_top_z = tz + t_hh

            z_tol = 0.06    # ± 6 cm contact tolerance
            xy_margin = 0.06

            resting: list[tuple] = []

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

                # Contact condition: other's bottom face rests on target's top face
                z_contact = abs(other_bottom_z - t_top_z) <= z_tol
                within_x = abs(ox - tx) <= t_hd + xy_margin
                within_y = abs(oy - ty) <= t_hw + xy_margin

                if z_contact and within_x and within_y:
                    bname = _extract_body_name(other_root)
                    resting.append((other_ann.__class__.__name__, bname, ox, oy, oz))

            if not resting:
                return f"No objects found resting on '{object_name}'."

            lines = [
                f"Objects supported by '{object_name}' ({len(resting)} found):"
            ]
            for cls_name, bname, ox, oy, oz in resting:
                lines.append(
                    f"  [{cls_name}]  body={bname}  pos=({ox:.3f}, {oy:.3f}, {oz:.3f})"
                )
            lines.append(
                f"Caution: these objects would be displaced if '{object_name}' is moved."
            )
            return "\n".join(lines)

        except Exception as e:
            return self._handle_error(e)

