"""Typed artifact models returned by agentic tools alongside their LLM-visible string."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from agentic_llmr.core.interfaces import Artifact


# ---------------------------------------------------------------------------
# Scene — inventory & taxonomy
# ---------------------------------------------------------------------------

class SceneObjectsArtifact(Artifact):
    objects: List[Dict[str, Any]]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "scene:all_objects", self.objects


class SemanticAnnotationsArtifact(Artifact):
    annotations: Dict[str, List[str]]  # cls_name → [body_names]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "scene:semantic_annotations", self.annotations


class FindObjectsArtifact(Artifact):
    type_name: str
    matches: List[Dict[str, Any]]  # [{body_name, position}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"type:{self.type_name}", self.matches


class ObjectTypeArtifact(Artifact):
    body_name: str
    type_hierarchy: List[str]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"obj:{self.body_name}", {"type_hierarchy": self.type_hierarchy}


class ClassifyByRoleArtifact(Artifact):
    surfaces: List[Dict[str, str]]
    articulated: List[Dict[str, str]]
    objects: List[Dict[str, str]]
    agent: List[Dict[str, str]]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "scene:role_map", self.model_dump()


# ---------------------------------------------------------------------------
# Scene — geometric & spatial
# ---------------------------------------------------------------------------

class PoseArtifact(Artifact):
    body_name: str
    position: Dict[str, float]   # {x, y, z}
    orientation: Dict[str, float]  # {x, y, z, w}

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"obj:{self.body_name}", {"pose": {"position": self.position, "orientation": self.orientation}}


class DimensionsArtifact(Artifact):
    body_name: str
    bounding_box_m: Dict[str, Any]   # {width, depth, height} or {error}
    volume_m3: Optional[float] = None
    center_of_mass: Optional[Dict[str, float]] = None

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"obj:{self.body_name}", {
            "dimensions": {
                "bounding_box_m": self.bounding_box_m,
                "volume_m3": self.volume_m3,
                "center_of_mass": self.center_of_mass,
            }
        }


class OrientationArtifact(Artifact):
    body_name: str
    status: str
    tilt_deg: float
    local_z_in_world: List[float]
    roll: float
    pitch: float
    yaw: float

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"obj:{self.body_name}", {"orientation": {
            "status": self.status,
            "tilt_deg": self.tilt_deg,
            "local_z_in_world": self.local_z_in_world,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
        }}


class ColorArtifact(Artifact):
    body_name: str
    colors: List[str]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"obj:{self.body_name}", {"color": self.colors}


class SpatialRelationArtifact(Artifact):
    object_name: str
    reference_name: str
    direction_str: str
    offset: Dict[str, float]   # {dx, dy, dz}
    distance_m: float

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"spatial:{self.object_name}→{self.reference_name}", {
            "direction": self.direction_str,
            "offset": self.offset,
            "distance_m": self.distance_m,
        }


class NearestObjectsArtifact(Artifact):
    reference: str
    nearest: List[Dict[str, Any]]  # [{cls_name, body_name, distance_m}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"nearest_to:{self.reference}", self.nearest


class ObjectsOnSurfaceArtifact(Artifact):
    surface_name: str
    objects: List[Dict[str, Any]]  # [{cls_name, body_name, position}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"on_surface:{self.surface_name}", self.objects


class SortBySizeArtifact(Artifact):
    type_name: str
    ranked: List[Dict[str, Any]]  # [{cls_name, body_names, volume_m3, position}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"size_rank:{self.type_name}", self.ranked


# ---------------------------------------------------------------------------
# Scene — structural & topological
# ---------------------------------------------------------------------------

class ArticulatedJointsArtifact(Artifact):
    object_name: str
    cls_name: str
    joints: List[Dict[str, Any]]  # [{name, type, position, limits}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"obj:{self.object_name}", {"joints": self.joints}


class ContainedItemsArtifact(Artifact):
    container: str
    items: List[Dict[str, Any]]  # [{cls_name, body_name, position}]
    method: str  # "semantic" | "geometric"

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"contained_in:{self.container}", self.items


# ---------------------------------------------------------------------------
# Scene — robot state
# ---------------------------------------------------------------------------

class RobotPoseArtifact(Artifact):
    body_name: str
    position: List[float]   # [x, y, z]
    orientation: List[float]  # [qx, qy, qz, qw]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "robot:base_pose", {
            "body_name": self.body_name,
            "position": self.position,
            "orientation": self.orientation,
        }


class EndEffectorPoseArtifact(Artifact):
    arm: str
    body_name: str
    position: List[float]
    orientation: List[float]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"robot:ee_pose_{self.arm}", {
            "body_name": self.body_name,
            "position": self.position,
            "orientation": self.orientation,
        }


class GripperStateArtifact(Artifact):
    arm: str
    gripper_type: str
    status: str   # "open" | "closed" | "partially open"
    width_m: Optional[float] = None
    range_pct: Optional[float] = None

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"robot:gripper_{self.arm}", self.model_dump()


class JointStatesArtifact(Artifact):
    joints: List[Dict[str, Any]]  # [{name, position}] — current positions only; limits live in kinematics

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "robot:joint_states", self.joints


class HeldObjectArtifact(Artifact):
    arm: str
    body_name: Optional[str]
    cls_name: Optional[str]
    position: Optional[List[float]]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"robot:held_{self.arm}", {
            "body_name": self.body_name,
            "cls_name": self.cls_name,
            "position": self.position,
        }


# ---------------------------------------------------------------------------
# Scene — collision, placement
# ---------------------------------------------------------------------------

class SceneCollisionsArtifact(Artifact):
    any_collision: bool
    contacts: List[Dict[str, Any]]  # [{body_a, body_b, distance_m}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "scene:collisions", {"any": self.any_collision, "contacts": self.contacts}


class FreeSpotsArtifact(Artifact):
    surface_name: str
    cls_name: str
    spots: List[Dict[str, float]]  # [{x, y, z}]
    occupancy_filtered: bool

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"free_spots:{self.surface_name}", self.spots


class WouldCollideArtifact(Artifact):
    object_name: str
    x: float
    y: float
    z: float
    collides: bool
    conflicts: List[str]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "scene:last_collision_check", {
            "object": self.object_name,
            "pose": {"x": self.x, "y": self.y, "z": self.z},
            "collides": self.collides,
            "conflicts": self.conflicts,
        }


# ---------------------------------------------------------------------------
# Scene — accessibility & causal
# ---------------------------------------------------------------------------

class IsAccessibleArtifact(Artifact):
    object_name: str
    accessible: bool
    blocker: Optional[str] = None
    reason: Optional[str] = None
    fix: Optional[str] = None

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"accessible:{self.object_name}", {
            "accessible": self.accessible,
            "blocker": self.blocker,
            "reason": self.reason,
            "fix": self.fix,
        }


class SupportingObjectArtifact(Artifact):
    object_name: str
    supported_by: Optional[str]   # cls_name or "floor" or None
    supporter_body: Optional[str]
    supporter_position: Optional[List[float]]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"supported_by:{self.object_name}", {
            "supported_by": self.supported_by,
            "body": self.supporter_body,
            "position": self.supporter_position,
        }


class SupportedByArtifact(Artifact):
    object_name: str
    resting_objects: List[Dict[str, Any]]  # [{cls_name, body_name, position}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"displaces:{self.object_name}", self.resting_objects


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

class ForwardKinematicsArtifact(Artifact):
    link_name: str
    position: Dict[str, float]    # {x, y, z}
    orientation: Dict[str, float]  # {x, y, z, w}

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"kin:fk:{self.link_name}", {
            "position": self.position,
            "orientation": self.orientation,
        }


class IKSolutionArtifact(Artifact):
    arm: str
    feasible: bool
    joint_positions: Dict[str, float]  # {joint_name: position_rad}
    error_message: Optional[str] = None

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"kin:ik:{self.arm}", {
            "feasible": self.feasible,
            "joint_positions": self.joint_positions,
            "error": self.error_message,
        }


class JointLimitsDetailedArtifact(Artifact):
    arm: str
    joints: List[Dict[str, Any]]  # [{name, position_limits, velocity_limits, acceleration_limits}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"kin:joint_limits:{self.arm}", self.joints


class SelfCollisionCheckArtifact(Artifact):
    any_self_collision: bool
    contacts: List[Dict[str, Any]]  # [{body_a, body_b, distance_m}]
    config_tested: Dict[str, float]  # joint_name → position

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "kin:self_collision_check", {
            "any_self_collision": self.any_self_collision,
            "contacts": self.contacts,
            "config_tested": self.config_tested,
        }


class ReachabilityArtifact(Artifact):
    results: List[Dict[str, Any]]  # [{arm, status, distance_m, max_reach_m, lateral_offset_m}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "kin:reachability", self.results


class ArmSuitabilityArtifact(Artifact):
    ranked: List[Dict[str, Any]]  # [{arm, score, criteria, ...}]
    recommendation: Optional[str]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "kin:preferred_arm", {
            "ranked": self.ranked,
            "recommendation": self.recommendation,
        }


class GraspPosesArtifact(Artifact):
    body_name: str
    approaches: List[Dict[str, str]]  # [{approach_direction, vertical_alignment}]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"kin:grasp_poses:{self.body_name}", self.approaches


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class AvailableActionsArtifact(Artifact):
    action_names: List[str]

    def to_fact_entry(self) -> Tuple[str, Any]:
        return "plan:available_actions", self.action_names


class ActionDocArtifact(Artifact):
    action_name: str
    documentation: str

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"plan:schema:{self.action_name}", self.documentation


class SimulationResultArtifact(Artifact):
    action_type: str
    success: bool
    message: str

    def to_fact_entry(self) -> Tuple[str, Any]:
        return f"plan:sim:{self.action_type}", {
            "success": self.success,
            "message": self.message,
        }
