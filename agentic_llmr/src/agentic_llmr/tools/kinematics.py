"""Kinematics tools — arm reachability, grasp poses, and robot state via Giskard."""

import logging
import math
import concurrent.futures
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from agentic_llmr.core.interfaces import AgenticTool
from agentic_llmr.platform.world import (
    get_active_world,
    compute_grasp_descriptions, get_bodies, get_arm_label,
    get_robot_base_body, symbol_display_name,
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
        "Call this first before any pick/place action to gate on whether each arm can reach at all. "
        "If both arms are Reachable, follow up with compare_arm_suitability to select the better one."
    )
    args_schema: Type[BaseModel] = ReachabilityInput

    def _run(self, target_pose: Dict[str, Any]) -> str:
        try:
            logger.debug(f"[Giskard Tool] Checking reachability for pose: {target_pose}")

            pos = target_pose.get("position", {})
            tx, ty, tz = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))

            world, robot_view = get_active_world()
            T_base_inv = _get_base_inv(world, robot_view)
            target_rf = _to_robot_frame(np.array([tx, ty, tz]), T_base_inv)

            arms = getattr(robot_view, "arms", [])
            if not arms:
                return "No arms found on robot."

            results = []
            for arm in arms:
                label = get_arm_label(arm, robot_view).upper()
                try:
                    T_sh = world.compute_forward_kinematics_np(world.root, arm.root)
                    shoulder_rf = _to_robot_frame(
                        np.array([T_sh[0, 3], T_sh[1, 3], T_sh[2, 3]]), T_base_inv
                    )
                except Exception:
                    results.append(f"{label}: could not read shoulder position")
                    continue

                delta = target_rf - shoulder_rf
                d = float(np.linalg.norm(delta))
                max_reach = _compute_arm_reach(arm, world)
                in_front = target_rf[0] > -0.2
                reachable = d <= max_reach and in_front
                status = "Reachable" if reachable else "Unreachable"
                # Signed lateral offset: positive → target is to the outer side of this arm
                lat = delta[1]
                results.append(
                    f"{label}: {status} "
                    f"(distance: {d:.3f}m / {max_reach:.3f}m, "
                    f"lateral offset: {lat:+.3f}m)"
                )

            return " | ".join(results)

        except Exception as e:
            return self._handle_error(e)


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
        "Call this after check_kinematic_reachability confirms both arms are Reachable — "
        "do not skip the reachability gate."
    )
    args_schema: Type[BaseModel] = ArmSuitabilityInput

    def _run(self, target_pose: Dict[str, Any], nearby_obstacles: List[str] = []) -> str:
        try:
            world, robot_view = get_active_world()
            arms = getattr(robot_view, "arms", [])
            if not arms:
                return "No arms found on robot."

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
            return _format_suitability_report(arm_results)

        except Exception as e:
            return self._handle_error(e)


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
        "Call this after get_object_orientation to select the most suitable approach. "
        "Requires the exact body_name (e.g., 'milk.stl') not a semantic class name."
    )
    args_schema: Type[BaseModel] = GraspPosesInput

    def _run(self, object_name: str) -> List[str]:
        try:
            logger.debug(f"[Giskard Tool] Getting grasp poses for: {object_name}")
            world, robot_view = get_active_world()

            target_body = None
            for body in get_bodies():
                if symbol_display_name(body) == object_name:
                    target_body = body
                    break

            if not target_body:
                return [f"Error: Object '{object_name}' not found in the active world."]

            arms = getattr(robot_view, "arms", [])
            if not arms:
                return ["Error: No arms found on robot."]
            manipulator = getattr(arms[0], "manipulator", None)
            if manipulator is None:
                return ["Error: No manipulator found on robot arm."]

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
                return [
                    "approach_direction: TOP,   vertical_alignment: NoAlignment",
                    "approach_direction: FRONT, vertical_alignment: NoAlignment",
                    "approach_direction: LEFT,  vertical_alignment: NoAlignment",
                    "approach_direction: RIGHT, vertical_alignment: NoAlignment",
                    "approach_direction: BACK,  vertical_alignment: NoAlignment",
                ]

            return [
                f"approach_direction: {g.approach_direction.name}, "
                f"vertical_alignment: {g.vertical_alignment.name}"
                for g in grasps
            ]

        except Exception as e:
            import traceback
            traceback.print_exc()
            return [self._handle_error(e)]
