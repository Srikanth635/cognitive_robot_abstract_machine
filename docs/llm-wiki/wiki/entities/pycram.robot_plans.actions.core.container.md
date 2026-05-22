---
id: pycram.robot_plans.actions.core.container
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/core/container.py
    lines: [1, 171]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.motions.container
  - pycram.robot_plans.motions.gripper
  - pycram.datastructures.grasp.GraspDescription
  - pycram.view_manager.ViewManager
  - sdt.world_description.world_entity.Body
  - pycram.datastructures.Context
  - pycram.pose_validator
  - pycram.querying.predicates
  - sdt.reasoning.robot_predicates
  - sdt.datastructures.GripperState
  - sdt.world_description.connections
  - pycram.datastructures.enums.Arms
used_by:
  - pycram.robot_plans.actions.composite
fields:
  object_designator:
    type: sdt.world_description.world_entity.Body
    description: The handle body to grasp (applies to both OpenAction and CloseAction).
  arm:
    type: pycram.datastructures.enums.Arms
    domain: [LEFT, RIGHT]
    description: Which arm to use.
  grasping_prepose_distance:
    type: float
    derived_from: ActionConfig.grasping_prepose_distance
    description: Pre-grasp standoff distance in metres; value read from ActionConfig at runtime.
status: stable
tags: [action, container, open, close, pre-post-condition]
last_ingest: 2026-05-19
---

_Two `ActionDescription` subclasses for manipulating articulated containers: `OpenAction` (grasp handle → drive open → release) and `CloseAction` (grasp handle → drive closed → release). Both include EQL pre/post-conditions._

## OpenAction

Grasps a container handle, drives the joint to its open position, then opens the gripper.

| Field | Type | Description |
|-------|------|-------------|
| `object_designator` | `Body` | Handle body to grasp |
| `arm` | `Arms` | Arm to use |
| `grasping_prepose_distance` | `float` | Pre-grasp standoff distance in metres (from `ActionConfig`) |

**execute()** sub-plan chain:
1. `GraspingAction(handle, arm, grasp_description)` — front-approach FRONT/NoAlignment grasp
2. `OpeningMotion(handle, arm)` — giskardpy `Open` goal drives joint to maximum
3. `MoveGripperMotion(GripperState.OPEN, arm, allow_gripper_collision=True)` — release

**pre_condition**: `GripperIsFree(manipulator)` (from `pycram.querying.predicates`) AND `reachability_validator(handle.global_pose, tool_frame, robot, test_world, full_body_controlled)`. Creates `test_world = deepcopy(context.world)` and instantiates a fresh `context.robot.from_world(test_world)` for the reachability test — full world + robot deepcopy.

**post_condition**: `is_body_in_gripper(handle, manipulator) > 0.9` (ray-sampling score from `sdt.reasoning.robot_predicates`) OR `allclose(handle.global_pose, tcp.global_pose, atol=0.03)` (3 cm fallback), AND `handle.get_first_parent_connection_of_type(ActiveConnection1DOF).position > 0.3`.

## CloseAction

Grasps a container handle, drives the joint near-closed (`goal_joint_state=0.01`), then opens the gripper.

| Field | Type | Description |
|-------|------|-------------|
| `object_designator` | `Body` | Handle body to grasp |
| `arm` | `Arms` | Arm to use |
| `grasping_prepose_distance` | `float` | Pre-grasp standoff (from `ActionConfig`) |

**execute()** sub-plan chain:
1. `GraspingAction(handle, arm, grasp_description)` — same FRONT/NoAlignment grasp
2. `ClosingMotion(handle, arm)` — giskardpy `Close(goal_joint_state=0.01)`
3. `MoveGripperMotion(GripperState.OPEN, arm, allow_gripper_collision=True)` — release

**post_condition**: `handle.get_first_parent_connection_of_type(ActiveConnection1DOF).position < 0.1`. No `pre_condition` — `CloseAction` does not check gripper occupancy before attempting to grasp.

## Design notes

**Hardcoded grasp approach:** Both actions always use `ApproachDirection.FRONT` + `VerticalAlignment.NoAlignment`. Unlike `PickUpAction`, they do not call `calculate_grasp_descriptions` — drawer/door handles are assumed to always be grasped from the front.

**Movable connection discovery:** Neither `OpenAction` nor `CloseAction` directly walks the kinematic tree to find the joint — they pass `object_designator` (the handle body) directly to `OpeningMotion`/`ClosingMotion`, which passes it as `environment_link` to giskardpy's `Open`/`Close` goal. It is the giskardpy goal that calls `get_first_parent_connection_of_type(ActiveConnection1DOF)` during `expand()`. The post_condition also calls this method directly on the handle body.

**Pre-condition world copy:** `OpenAction.pre_condition` creates `deepcopy(context.world)` and `context.robot.from_world(test_world)` to run `reachability_validator` — a full world AND robot deepcopy, not just the state save/restore used by `pose_sequence_reachability_validator`.

## Related

**Uses:** [[pycram.robot_plans.ActionDescription]], [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.motions.container]], [[pycram.robot_plans.motions.gripper]], [[pycram.datastructures.grasp.GraspDescription]], [[pycram.view_manager.ViewManager]], [[sdt.world_description.world_entity.Body]], [[pycram.datastructures.Context]], [[pycram.pose_validator]], [[pycram.querying.predicates]], [[sdt.reasoning.robot_predicates]], [[sdt.world_description.connections]]

**Used by:** [[pycram.robot_plans.actions.composite]]

## Provenance

- `pycram/src/pycram/robot_plans/actions/core/container.py` lines 1–171 (commit `0528d8cf3`) — `OpenAction`, `CloseAction`.
