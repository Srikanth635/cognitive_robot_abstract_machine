"""Kinematics tools — arm reachability, grasp poses, and robot state via Giskard."""

import logging
import math
import concurrent.futures
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool
from agentic_llmr.tools.artifacts import (
    ReachabilityArtifact,
    ArmSuitabilityArtifact,
    GraspPosesArtifact,
    ForwardKinematicsArtifact,
    IKSolutionArtifact,
    JointLimitsDetailedArtifact,
    SelfCollisionCheckArtifact,
)
from agentic_llmr.platform.world import (
    get_active_world,
    compute_grasp_descriptions, compute_ik_bridge,
    get_bodies, get_arm_label,
    get_robot_base_body, symbol_display_name, find_body_by_name,
)
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

_GRASP_TIMEOUT = 6.0       # seconds before get_grasp_poses gives up
_PAN_HALF_RANGE = math.pi / 2   # ±90° — conservative typical shoulder pan range
_ELEV_MIN = math.radians(-45)   # minimum elevation arm is expected to reach
_ELEV_MAX = math.radians(90)    # maximum elevation arm is expected to reach
_CLEARANCE_SAFE = 0.40     # metres — clearance above this scores 1.0
_CLEARANCE_ZERO = 0.05     # metres — clearance below this scores 0.0


# ── Shared geometry helpers ────────────────────────────────────────────────────


def _compute_arm_reach(arm: Any, world: Any) -> float:
    """Sum of consecutive link lengths along the arm's kinematic chain."""
    try:
        bodies = [
            e for e in arm.kinematic_structure_entities
            if "Body" in {cls.__name__ for cls in type(e).__mro__}
        ]
        if len(bodies) < 2:
            return 0.75
        total = 0.0
        for i in range(len(bodies) - 1):
            T1 = world.compute_forward_kinematics_np(world.root, bodies[i])
            T2 = world.compute_forward_kinematics_np(world.root, bodies[i + 1])
            total += float(np.linalg.norm(T2[:3, 3] - T1[:3, 3]))
        return total
    except Exception:
        return 0.75


def _get_base_inv(world: Any, robot_view: Any) -> Optional[np.ndarray]:
    """Return the inverse base-frame transform, or None if unavailable."""
    base_body = get_robot_base_body(robot_view)
    if base_body is None:
        return None
    try:
        return np.linalg.inv(
            world.compute_forward_kinematics_np(world.root, base_body)
        )
    except Exception:
        return None


def _to_robot_frame(pt_world: np.ndarray, T_base_inv: Optional[np.ndarray]) -> np.ndarray:
    """Transform a world-frame 3-vector into the robot base frame."""
    if T_base_inv is None:
        return pt_world.copy()
    p = T_base_inv @ np.array([pt_world[0], pt_world[1], pt_world[2], 1.0])
    return np.array([float(p[0]), float(p[1]), float(p[2])])


def _segment_to_point_dist(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Minimum Euclidean distance from point p to line segment [a, b]."""
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-9:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_sq))
    return float(np.linalg.norm(p - (a + t * ab)))


# ── CheckKinematicReachabilityTool ─────────────────────────────────────────────


class ReachabilityInput(BaseModel):
    """Input schema for CheckReachabilityTool."""

    target_pose: Dict[str, Any] = Field(
        description="Dict with 'position' (x, y, z) and 'orientation' (x, y, z, w)."
    )


class CheckReachabilityTool(AgenticTool):
    """Report Reachable/Unreachable for each arm given a target 3D world-frame pose."""

    name: str = "check_kinematic_reachability"
    description: str = (
        "Binary reachability check for a target 3D pose. "
        "Reports Reachable/Unreachable, shoulder-to-target distance, and signed lateral offset "
        "(positive = target is to the arm's outer side) for each arm. "
        "Answers whether a given world-frame target lies within an arm's workspace."
    )
    args_schema: Type[BaseModel] = ReachabilityInput

    def _query(self, target_pose: Dict[str, Any]) -> ReachabilityArtifact:
        logger.debug(f"[Giskard Tool] Checking reachability for pose: {target_pose}")

        pos = target_pose.get("position", {})
        tx, ty, tz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))

        world, robot_view = get_active_world()
        T_base_inv = _get_base_inv(world, robot_view)
        target_rf = _to_robot_frame(np.array([tx, ty, tz]), T_base_inv)

        arms = getattr(robot_view, "arms", [])
        if not arms:
            raise RuntimeError("No arms found on robot.")

        results = []
        for arm in arms:
            label = get_arm_label(arm, robot_view).upper()
            try:
                T_sh = world.compute_forward_kinematics_np(world.root, arm.root)
                shoulder_rf = _to_robot_frame(
                    np.array([T_sh[0, 3], T_sh[1, 3], T_sh[2, 3]]), T_base_inv
                )
            except Exception:
                results.append({"arm": label, "status": "error", "distance_m": None,
                                 "max_reach_m": None, "lateral_offset_m": None})
                continue

            delta = target_rf - shoulder_rf
            d = float(np.linalg.norm(delta))
            max_reach = _compute_arm_reach(arm, world)
            in_front = target_rf[0] > -0.2
            reachable = d <= max_reach and in_front
            results.append({
                "arm": label,
                "status": "Reachable" if reachable else "Unreachable",
                "distance_m": round(d, 3),
                "max_reach_m": round(max_reach, 3),
                "lateral_offset_m": round(float(delta[1]), 3),
            })

        return ReachabilityArtifact(results=results)

    def _format(self, artifact: ReachabilityArtifact) -> str:
        parts = []
        for r in artifact.results:
            arm = r["arm"]
            if r["status"] == "error":
                parts.append(f"{arm}: could not read shoulder position")
            else:
                parts.append(
                    f"{arm}: {r['status']} "
                    f"(distance: {r['distance_m']:.3f}m / {r['max_reach_m']:.3f}m, "
                    f"lateral offset: {r['lateral_offset_m']:+.3f}m)"
                )
        return " | ".join(parts)


# ── CompareArmSuitabilityTool helpers ─────────────────────────────────────────


def _collect_robot_body_names(robot_view: Any, arms: List[Any]) -> set:
    """Return display names of all bodies that belong to the robot (used to exclude them from obstacles)."""
    names: set = set()
    try:
        for arm in arms:
            for entity in getattr(arm, "kinematic_structure_entities", []):
                if "Body" in {cls.__name__ for cls in type(entity).__mro__}:
                    names.add(symbol_display_name(entity))
        base_body = get_robot_base_body(robot_view)
        if base_body is not None:
            names.add(symbol_display_name(base_body))
    except Exception:
        pass
    return names


def _resolve_obstacle_positions(
    nearby_obstacles: List[str],
    robot_body_names: set,
    world: Any,
    T_base_inv: Optional[np.ndarray],
) -> List[tuple]:
    """Resolve obstacle body names to robot-frame positions, skipping robot parts."""
    positions: List[tuple] = []
    for name in nearby_obstacles:
        if name in robot_body_names:
            continue
        for body in get_bodies():
            if symbol_display_name(body) == name:
                try:
                    T = world.compute_forward_kinematics_np(world.root, body)
                    obs_rf = _to_robot_frame(
                        np.array([T[0, 3], T[1, 3], T[2, 3]]), T_base_inv
                    )
                    positions.append((name, obs_rf))
                except Exception:
                    pass
                break
    return positions


def _score_lateral_alignment(delta: np.ndarray, max_reach: float) -> tuple:
    """Score how well-aligned the shoulder is laterally with the target.

    Returns (score, lat_offset) where score ∈ [0, 1] and lat_offset is signed y-distance.
    Smaller lateral offset → less shoulder-pan rotation needed → higher score.
    """
    lat_offset = abs(delta[1])
    score = max(0.0, 1.0 - lat_offset / max_reach)
    return score, delta[1]


def _score_joint_limit_margin(delta: np.ndarray) -> tuple:
    """Score how far the required shoulder angles are from joint limits.

    Returns (score, pan_angle_rad, elev_angle_rad) where score ∈ [0, 1].
    Higher score = more margin away from joint limits = safer configuration.
    """
    horiz_dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2)
    pan_angle = math.atan2(delta[1], max(horiz_dist, 1e-3))
    pan_margin = max(0.0, (_PAN_HALF_RANGE - abs(pan_angle)) / _PAN_HALF_RANGE)

    elev_angle = math.atan2(delta[2], max(horiz_dist, 1e-3))
    elev_centre = (_ELEV_MAX + _ELEV_MIN) / 2.0
    elev_half = (_ELEV_MAX - _ELEV_MIN) / 2.0
    elev_margin = max(0.0, 1.0 - abs(elev_angle - elev_centre) / elev_half)

    score = 0.6 * pan_margin + 0.4 * elev_margin
    return score, pan_angle, elev_angle


def _score_obstacle_clearance(
    shoulder_rf: np.ndarray,
    target_rf: np.ndarray,
    obstacle_positions: List[tuple],
) -> tuple:
    """Score the minimum clearance from the straight shoulder→target path to each obstacle.

    Returns (score, min_clearance_m). min_clearance_m is None when no obstacles are present.
    Approximates the arm's swept path as a straight line segment.
    """
    if not obstacle_positions:
        return 1.0, None
    min_clearance = min(
        _segment_to_point_dist(shoulder_rf, target_rf, obs_rf)
        for _, obs_rf in obstacle_positions
    )
    score = max(
        0.0,
        min(1.0, (min_clearance - _CLEARANCE_ZERO) / (_CLEARANCE_SAFE - _CLEARANCE_ZERO)),
    )
    return score, min_clearance


def _format_suitability_report(arm_results: List[tuple]) -> str:
    """Format arm suitability scores and recommendation into a human-readable string."""
    lines = ["Arm suitability comparison:"]
    for rank, (label, score, detail) in enumerate(arm_results, 1):
        if score is None:
            lines.append(f"  #{rank} {label}: {detail}")
        else:
            lines.append(f"  #{rank} {label}  [score {score:.2f}]")
            for k, v in detail.items():
                if k != "total":
                    lines.append(f"       {k:14s}: {v}")

    best_feasible = next(
        ((lbl, sc) for lbl, sc, _ in arm_results if sc is not None), None
    )
    if best_feasible:
        lines.append(f"\nRecommendation: {best_feasible[0]} arm  (score {best_feasible[1]:.2f})")
    else:
        lines.append("\nNo reachable arm found — consider navigating closer.")
    return "\n".join(lines)


# ── CompareArmSuitabilityTool ──────────────────────────────────────────────────


class ArmSuitabilityInput(BaseModel):
    """Input schema for CompareArmSuitabilityTool."""

    target_pose: Dict[str, Any] = Field(
        description="Dict with 'position' (x, y, z) and 'orientation' (x, y, z, w)."
    )
    nearby_obstacles: List[str] = Field(
        default_factory=list,
        description=(
            "Body names of objects that may obstruct arm paths "
            "(e.g. from get_nearest_objects or get_objects_on_surface). "
            "Pass an empty list when no obstacle context is available."
        ),
    )


class CompareArmSuitabilityTool(AgenticTool):
    """Score and rank both arms by lateral alignment, joint-limit margin, and obstacle clearance."""

    name: str = "compare_arm_suitability"
    description: str = (
        "Score and rank all robot arms for a given target pose on three criteria:\n"
        "  1. Lateral alignment  — signed y-offset from shoulder to target; "
        "arm whose shoulder is laterally closer needs less shoulder-pan rotation.\n"
        "  2. Joint limit margin — estimated shoulder-pan and elevation angles vs. "
        "joint ranges; higher margin = safer, more dexterous configuration.\n"
        "  3. Obstacle clearance — minimum distance from each arm's straight-line "
        "shoulder→target path to each named obstacle body.\n"
        "Returns per-criterion scores, a composite score, and a final recommendation. "
        "Use it to choose between arms that can reach a target."
    )
    args_schema: Type[BaseModel] = ArmSuitabilityInput

    def _query(self, target_pose: Dict[str, Any], nearby_obstacles: List[str] = []) -> ArmSuitabilityArtifact:
        world, robot_view = get_active_world()
        arms = getattr(robot_view, "arms", [])
        if not arms:
            raise RuntimeError("No arms found on robot.")

        pos = target_pose.get("position", {})
        tx, ty, tz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
        T_base_inv = _get_base_inv(world, robot_view)
        target_rf = _to_robot_frame(np.array([tx, ty, tz]), T_base_inv)

        robot_body_names = _collect_robot_body_names(robot_view, arms)
        obstacle_positions = _resolve_obstacle_positions(
            nearby_obstacles, robot_body_names, world, T_base_inv
        )

        arm_results = []
        for arm in arms:
            label = get_arm_label(arm, robot_view).upper()
            try:
                T_sh = world.compute_forward_kinematics_np(world.root, arm.root)
                shoulder_rf = _to_robot_frame(
                    np.array([T_sh[0, 3], T_sh[1, 3], T_sh[2, 3]]), T_base_inv
                )
            except Exception:
                arm_results.append((label, None, "Could not read shoulder position."))
                continue

            max_reach = _compute_arm_reach(arm, world)
            delta = target_rf - shoulder_rf
            d = float(np.linalg.norm(delta))

            if not (d <= max_reach and target_rf[0] > -0.2):
                arm_results.append(
                    (label, None, f"Unreachable (d={d:.3f}m, max={max_reach:.3f}m)")
                )
                continue

            lateral_score, lat_signed = _score_lateral_alignment(delta, max_reach)
            joint_score, pan_angle, elev_angle = _score_joint_limit_margin(delta)
            clearance_score, min_clearance = _score_obstacle_clearance(
                shoulder_rf, target_rf, obstacle_positions
            )

            if obstacle_positions:
                w_lat, w_jnt, w_clr = 0.35, 0.35, 0.30
            else:
                w_lat, w_jnt, w_clr = 0.50, 0.50, 0.00

            total = w_lat * lateral_score + w_jnt * joint_score + w_clr * clearance_score
            clearance_str = (
                f"{min_clearance:.3f}m" if min_clearance is not None else "N/A (no obstacles)"
            )
            arm_results.append((label, total, {
                "lateral":      f"{lateral_score:.2f}  (y-offset {lat_signed:+.3f}m)",
                "joint_margin": (
                    f"{joint_score:.2f}  "
                    f"(pan {math.degrees(pan_angle):+.1f}°, "
                    f"elev {math.degrees(elev_angle):+.1f}°)"
                ),
                "clearance":    f"{clearance_score:.2f}  (min dist {clearance_str})",
                "total":        f"{total:.2f}",
            }))

        arm_results.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))

        best_feasible = next(
            ((lbl, sc) for lbl, sc, _ in arm_results if sc is not None), None
        )
        recommendation = f"{best_feasible[0]} arm" if best_feasible else None

        ranked = []
        for label, score, detail in arm_results:
            ranked.append({"arm": label, "score": score, "criteria": detail})

        return ArmSuitabilityArtifact(ranked=ranked, recommendation=recommendation)

    def _format(self, artifact: ArmSuitabilityArtifact) -> str:
        # Rebuild the arm_results tuple list for the existing formatter
        arm_results = []
        for entry in artifact.ranked:
            arm_results.append((entry["arm"], entry["score"], entry["criteria"]))
        return _format_suitability_report(arm_results)


# ── GetGraspPosesTool ──────────────────────────────────────────────────────────


class GraspPosesInput(BaseModel):
    """Input schema for GetGraspPosesTool."""

    object_name: str = Field(
        description=(
            "The exact body_name of the object (e.g., 'milk.stl'). "
            "Must match the body_name from the scene, not a semantic class name like 'Milk'."
        )
    )


class GetGraspPosesTool(AgenticTool):
    """Return valid grasp approach direction labels (TOP/FRONT/LEFT/RIGHT/BACK) for a target body."""

    name: str = "get_grasp_poses"
    description: str = (
        "Get valid grasp approach direction labels for an object: TOP, FRONT, LEFT, RIGHT, BACK. "
        "Returns discrete direction labels and vertical alignment (NoAlignment/TOP/BOTTOM) — "
        "not full 6-DOF poses. "
        "Requires the exact body_name (e.g., 'milk.stl') not a semantic class name."
    )
    args_schema: Type[BaseModel] = GraspPosesInput

    def _query(self, object_name: str) -> GraspPosesArtifact:
        logger.debug(f"[Giskard Tool] Getting grasp poses for: {object_name}")
        world, robot_view = get_active_world()

        target_body = None
        for body in get_bodies():
            if symbol_display_name(body) == object_name:
                target_body = body
                break

        if not target_body:
            raise ValueError(f"Object '{object_name}' not found in the active world.")

        arms = getattr(robot_view, "arms", [])
        if not arms:
            raise RuntimeError("No arms found on robot.")
        manipulator = getattr(arms[0], "manipulator", None)
        if manipulator is None:
            raise RuntimeError("No manipulator found on robot arm.")

        T = world.compute_forward_kinematics_np(world.root, target_body)
        x, y, z = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat()

        def _compute():
            return compute_grasp_descriptions(
                manipulator,
                (float(x), float(y), float(z)),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                world.root,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_compute)
            try:
                grasps = future.result(timeout=_GRASP_TIMEOUT)
            except concurrent.futures.TimeoutError:
                grasps = None

        if grasps is None:
            approaches = [
                {"approach_direction": "TOP",   "vertical_alignment": "NoAlignment"},
                {"approach_direction": "FRONT", "vertical_alignment": "NoAlignment"},
                {"approach_direction": "LEFT",  "vertical_alignment": "NoAlignment"},
                {"approach_direction": "RIGHT", "vertical_alignment": "NoAlignment"},
                {"approach_direction": "BACK",  "vertical_alignment": "NoAlignment"},
            ]
        else:
            approaches = [
                {
                    "approach_direction": g.approach_direction.name,
                    "vertical_alignment": g.vertical_alignment.name,
                }
                for g in grasps
            ]

        return GraspPosesArtifact(body_name=object_name, approaches=approaches)

    def _format(self, artifact: GraspPosesArtifact) -> str:
        return "\n".join(
            f"approach_direction: {a['approach_direction']}, "
            f"vertical_alignment: {a['vertical_alignment']}"
            for a in artifact.approaches
        )


# ── Shared helpers for new tools ───────────────────────────────────────────────


def _find_arm_by_label(robot_view: Any, label: str) -> Optional[Any]:
    """Return the arm annotation whose label matches 'left' or 'right' (case-insensitive)."""
    arms = getattr(robot_view, "arms", [])
    for arm in arms:
        if get_arm_label(arm, robot_view).lower() == label.lower():
            return arm
    return None


def _extract_body_name(body: Any) -> str:
    """Return display name for a body, tolerating various name wrapper types."""
    return symbol_display_name(body) or repr(body)


def _iter_arm_controlled_connections(arm_annotation: Any):
    """Yield (conn, dof, joint_name) for every controlled connection in an arm."""
    seen = set()
    for js in getattr(arm_annotation, "joint_states", []):
        for conn in getattr(js, "connections", []):
            conn_id = id(conn)
            if conn_id in seen:
                continue
            seen.add(conn_id)
            if not getattr(conn, "is_controlled", False):
                continue
            if not (hasattr(conn, "position") and hasattr(conn, "dof")):
                continue
            name_obj = getattr(conn, "name", None)
            jname = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
            yield conn, conn.dof, jname


# ── ComputeForwardKinematicsTool ───────────────────────────────────────────────


class FKInput(BaseModel):
    link_name: str = Field(
        description=(
            "Exact name of the robot kinematic-tree link to compute FK for, e.g. "
            "'r_gripper_tool_frame' (hand), 'r_wrist_roll_link' (wrist), "
            "'r_elbow_flex_link' (elbow), 'r_forearm_link', 'r_upper_arm_link'. "
            "Left-arm links use the 'l_' prefix."
        )
    )


class ComputeForwardKinematicsTool(AgenticTool):
    """Compute the current world-frame pose of any named link in the kinematic tree."""

    name: str = "compute_forward_kinematics"
    description: str = (
        "Compute the current world-frame position and orientation of a named robot link. "
        "Returns the 6-DoF pose (position + quaternion) of the requested body in world coordinates. "
        "Use this to confirm where a specific wrist, elbow, or tool-frame link actually is right now. "
        "Requires the exact body_name (not a semantic class name)."
    )
    args_schema: Type[BaseModel] = FKInput

    def _run(self, link_name: str):
        try:
            world, _ = get_active_world()
            body = find_body_by_name(link_name)
            if body is None:
                return f"Link '{link_name}' not found in the world.", None
            T = world.compute_forward_kinematics_np(world.root, body)
            pos = {
                "x": round(float(T[0, 3]), 5),
                "y": round(float(T[1, 3]), 5),
                "z": round(float(T[2, 3]), 5),
            }
            quat = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
            ori = {
                "x": round(float(quat[0]), 5),
                "y": round(float(quat[1]), 5),
                "z": round(float(quat[2]), 5),
                "w": round(float(quat[3]), 5),
            }
            artifact = ForwardKinematicsArtifact(link_name=link_name, position=pos, orientation=ori)
            content = (
                f"FK of '{link_name}':\n"
                f"  position:    ({pos['x']:.4f}, {pos['y']:.4f}, {pos['z']:.4f})\n"
                f"  orientation (xyzw): ({ori['x']:.4f}, {ori['y']:.4f}, {ori['z']:.4f}, {ori['w']:.4f})"
            )
            return content, artifact
        except Exception as e:
            return self._handle_error(e), None


# ── SolveInverseKinematicsTool ─────────────────────────────────────────────────


class IKInput(BaseModel):
    arm: str = Field(description="Arm to solve IK for: 'left' or 'right'.")
    target_pose: Dict[str, Any] = Field(
        description=(
            "Target 6-DoF pose in world frame. "
            "Dict with 'position' ({x, y, z}) and 'orientation' ({x, y, z, w})."
        )
    )


class SolveInverseKinematicsTool(AgenticTool):
    """Compute a joint configuration that places an arm's tool frame at a target pose."""

    name: str = "solve_inverse_kinematics"
    description: str = (
        "Compute the joint configuration that places the arm's tool frame at the given "
        "world-frame 6-DoF pose. Returns the joint positions (in radians) if a solution exists. "
        "Distinct from check_kinematic_reachability (binary yes/no): this returns the actual joint "
        "values needed to verify joint-limit feasibility or pre-validate a motion before executing it. "
        "Follow up with check_self_collision_at_config to confirm the solution is collision-free."
    )
    args_schema: Type[BaseModel] = IKInput

    def _run(self, arm: str, target_pose: Dict[str, Any]):
        try:
            world, robot_view = get_active_world()
            arm_annotation = _find_arm_by_label(robot_view, arm)
            if arm_annotation is None:
                return f"Arm '{arm}' not found on robot.", None

            root_body = getattr(arm_annotation, "root", None)
            manipulator = getattr(arm_annotation, "manipulator", None)
            tip_body = (
                getattr(manipulator, "tool_frame", None)
                or getattr(arm_annotation, "tip", None)
            )
            if root_body is None or tip_body is None:
                return (
                    f"Could not resolve root/tip bodies for arm '{arm}'. "
                    "Check that the robot model has a manipulator with a tool_frame.", None
                )

            pos = target_pose.get("position", {})
            ori = target_pose.get("orientation", {})
            x, y, z = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
            qx = float(ori.get("x", 0))
            qy = float(ori.get("y", 0))
            qz = float(ori.get("z", 0))
            qw = float(ori.get("w", 1))

            joint_positions = compute_ik_bridge(world, root_body, tip_body, x, y, z, qx, qy, qz, qw)

            artifact = IKSolutionArtifact(arm=arm, feasible=True, joint_positions=joint_positions)
            lines = [f"IK solution for '{arm}' arm (tool frame → target pose):"]
            for jname, jpos in joint_positions.items():
                lines.append(f"  {jname}: {jpos:.4f} rad")
            return "\n".join(lines), artifact

        except Exception as e:
            msg = str(e)
            artifact = IKSolutionArtifact(arm=arm, feasible=False, joint_positions={}, error_message=msg)
            return f"IK failed for '{arm}' arm: {msg}", artifact


# ── GetJointLimitsDetailedTool ─────────────────────────────────────────────────


class JointLimitsInput(BaseModel):
    arm: str = Field(description="Arm to query: 'left' or 'right'.")


class GetJointLimitsDetailedTool(AgenticTool):
    """Return position, velocity, and acceleration limits for every joint in an arm."""

    name: str = "get_joint_limits_detailed"
    description: str = (
        "Get position, velocity, and acceleration limits for each controllable joint in the "
        "specified arm. Returns per-joint: position range [min, max] in radians, velocity limit "
        "(rad/s), and acceleration limit (rad/s²) where defined in the robot model. "
        "Use this to validate whether a planned joint configuration or motion profile is within "
        "hardware constraints. It also reports the exact joint names for an arm."
    )
    args_schema: Type[BaseModel] = JointLimitsInput

    def _run(self, arm: str):
        try:
            _, robot_view = get_active_world()
            arm_annotation = _find_arm_by_label(robot_view, arm)
            if arm_annotation is None:
                return f"Arm '{arm}' not found on robot.", None

            joints = []
            rows = []

            def _safe_float(limits_map, attr: str) -> Optional[float]:
                try:
                    v = getattr(limits_map, attr, None)
                    return round(float(v), 5) if v is not None else None
                except Exception:
                    return None

            for conn, dof, jname in _iter_arm_controlled_connections(arm_annotation):
                lims = dof.limits
                lo = lims.lower if lims else None
                hi = lims.upper if lims else None

                pos_lo = _safe_float(lo, "position")
                pos_hi = _safe_float(hi, "position")
                vel_lo = _safe_float(lo, "velocity")
                vel_hi = _safe_float(hi, "velocity")
                acc_lo = _safe_float(lo, "acceleration")
                acc_hi = _safe_float(hi, "acceleration")

                joints.append({
                    "name": jname,
                    "position_limits": [pos_lo, pos_hi],
                    "velocity_limits": [vel_lo, vel_hi],
                    "acceleration_limits": [acc_lo, acc_hi],
                })

                pos_str = (
                    f"[{pos_lo:.3f}, {pos_hi:.3f}] rad"
                    if (pos_lo is not None and pos_hi is not None) else "unlimited"
                )
                vel_str = f"±{vel_hi:.3f} rad/s" if vel_hi is not None else "N/A"
                acc_str = f"±{acc_hi:.3f} rad/s²" if acc_hi is not None else "N/A"
                rows.append(f"  {jname}:  pos={pos_str}  vel={vel_str}  acc={acc_str}")

            if not joints:
                return f"No controllable joints found for arm '{arm}'.", None

            artifact = JointLimitsDetailedArtifact(arm=arm, joints=joints)
            content = f"Joint limits for '{arm}' arm ({len(joints)} joints):\n" + "\n".join(rows)
            return content, artifact

        except Exception as e:
            return self._handle_error(e), None


# ── CheckSelfCollisionAtConfigTool ─────────────────────────────────────────────


class SelfCollisionInput(BaseModel):
    joint_positions: Dict[str, float] = Field(
        description=(
            "Joint name → target position in radians. "
            "Joint names must exactly match those reported by get_joint_limits_detailed "
            "(e.g. {'r_shoulder_pan_joint': 1.2, 'r_elbow_flex_joint': -0.5}). "
            "Only the joints you specify are moved; the rest stay at their current positions."
        )
    )


class CheckSelfCollisionAtConfigTool(AgenticTool):
    """Check whether the robot self-collides at a given joint configuration."""

    name: str = "check_self_collision_at_config"
    description: str = (
        "Temporarily move the robot to a specified joint configuration in simulation, "
        "check for robot body-to-body (self) collisions, then restore the original state. "
        "Returns which robot body pairs intersect and by how much. "
        "Only robot-to-robot contacts are reported — scene-object contacts are excluded. "
        "Answers whether a given joint configuration is self-collision-free."
    )
    args_schema: Type[BaseModel] = SelfCollisionInput

    def _run(self, joint_positions: Dict[str, float]):
        try:
            world, robot_view = get_active_world()

            arms = getattr(robot_view, "arms", [])
            robot_body_names = _collect_robot_body_names(robot_view, arms)

            # Resolve joint names → (dof_id, target_position)
            dof_updates: Dict = {}  # dof.id → float
            unmatched = []

            parts = []
            for attr in ("arms", "neck", "torso", "base", "drive"):
                val = getattr(robot_view, attr, None)
                if val is None:
                    continue
                parts.extend(val if isinstance(val, list) else [val])

            for jname, jpos in joint_positions.items():
                matched = False
                for part in parts:
                    for conn, dof, cname in _iter_arm_controlled_connections(part):
                        if cname == jname:
                            dof_updates[dof.id] = float(jpos)
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    unmatched.append(jname)

            if unmatched:
                return (
                    f"Joint(s) not found: {unmatched}. "
                    "Call get_joint_limits_detailed for the arm to get its exact joint names, "
                    "then retry with those.", None
                )

            # Snapshot, apply config, check, restore
            saved = world.state._data.copy()
            contacts = []
            any_self_collision = False
            try:
                for dof_id, pos in dof_updates.items():
                    world.state[dof_id].position = pos
                world.notify_state_change()

                result = world.collision_manager.compute_collisions()
                if hasattr(result, "contacts") and result.contacts:
                    for contact in result.contacts:
                        name_a = _extract_body_name(contact.body_a)
                        name_b = _extract_body_name(contact.body_b)
                        if name_a in robot_body_names and name_b in robot_body_names:
                            any_self_collision = True
                            contacts.append({
                                "body_a": name_a,
                                "body_b": name_b,
                                "distance_m": round(float(contact.distance), 5),
                            })
            finally:
                world.state._data[:] = saved
                world.notify_state_change()

            artifact = SelfCollisionCheckArtifact(
                any_self_collision=any_self_collision,
                contacts=contacts,
                config_tested=joint_positions,
            )

            if any_self_collision:
                lines = [f"Self-collision detected ({len(contacts)} contact(s)):"]
                for c in contacts:
                    lines.append(
                        f"  {c['body_a']} ↔ {c['body_b']}  "
                        f"(penetration: {c['distance_m']:.4f} m)"
                    )
                return "\n".join(lines), artifact

            return (
                f"No self-collision at the given config "
                f"({len(joint_positions)} joint(s) tested). Config is safe.",
                artifact,
            )

        except Exception as e:
            return self._handle_error(e), None
