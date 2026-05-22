---
id: pycram.robot_plans.actions.core.robot_body
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/core/robot_body.py
    lines: [1, 241]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.motions.robot_body
  - pycram.robot_plans.motions.gripper
  - pycram.view_manager.ViewManager
  - sdt.robots.abstract_robot.AbstractRobot
  - pycram.datastructures.Context
  - sdt.datastructures.GripperState
  - pycram.datastructures.enums.Arms
  - pycram.datastructures.enums.AxisIdentifier
used_by:
  - pycram.robot_plans.actions.composite
fields:
  MoveTorsoAction:
    torso_state:
      type: sdt.datastructures.joint_state.JointState
      description: Named torso configuration (HIGH, LOW, etc.); resolved via robot.torso.get_joint_state_by_type at runtime.
  SetGripperAction:
    gripper:
      type: pycram.datastructures.enums.Arms
      domain: [LEFT, RIGHT, BOTH]
      description: Which gripper(s) to move.
    motion:
      type: sdt.datastructures.GripperState
      domain: [OPEN, CLOSE]
      description: Open or close the gripper.
  ParkArmsAction:
    arm:
      type: pycram.datastructures.enums.Arms
      description: Which arm(s) to park; BOTH parks both simultaneously.
  CarryAction:
    arm:
      type: pycram.datastructures.enums.Arms
      description: Arm to carry with.
    align:
      type: bool
      default: false
      description: If true, aligns the end-effector axis to a specified frame axis during carry motion.
  FollowToolCenterPointPathAction:
    target_locations:
      type: sdt.spatial_types.Pose
      description: Ordered sequence of TCP target poses (PoseTrajectory — list of Poses).
    arm:
      type: pycram.datastructures.enums.Arms
      description: Arm whose TCP follows the trajectory.
status: stable
tags: [action, torso, gripper, park, joints, robot-body]
last_ingest: 2026-05-19
---

_Five `ActionDescription` subclasses for whole-body configuration management: `MoveTorsoAction`, `SetGripperAction`, `ParkArmsAction`, `CarryAction`, `FollowToolCenterPointPathAction`. All wrap `MoveJointsMotion` or `MoveTCPWaypointsMotion`._

## MoveTorsoAction

Moves the robot torso to a named state (`TorsoState` enum: HIGH, LOW, etc.).

`execute()`: `robot.torso.get_joint_state_by_type(torso_state)` → `MoveJointsMotion(names, values)`.

`post_condition`: `joint_state.is_achieved()` — verifies the torso reached the target.

## SetGripperAction

Opens or closes both grippers or a single gripper.

| Field | Type | Description |
|-------|------|-------------|
| `gripper` | `Arms` | Which gripper(s) to move |
| `motion` | `GripperState` | OPEN or CLOSE |

`execute()`: for `Arms.BOTH`, runs two `MoveGripperMotion` calls sequentially. Otherwise a single call. No post_condition.

## ParkArmsAction

Parks one or both arms to a safe carry configuration.

| Field | Type | Description |
|-------|------|-------------|
| `arm` | `Arms` | Which arm(s) to park |

`execute()`:
1. `ViewManager.get_all_arm_views(arm, robot)` — resolves the arm chain(s).
2. `arm_chain.get_joint_state_by_type(StaticJointState.PARK)` — gets named park positions.
3. `MoveJointsMotion(names, values)` — drives joints to park.

Commonly called before and after pick/place/transport to clear occlusion and reduce collision risk.

## CarryAction

Parks arms and optionally aligns the end-effector axis with a specified frame axis. Used for carrying objects in a stable pose while navigating.

| Field | Type | Description |
|-------|------|-------------|
| `arm` | `Arms` | Arm to carry with |
| `align` | `bool` | If `True`, pass alignment axes to `MoveJointsMotion` |
| `tip_link` / `tip_axis` | `str` / `AxisIdentifier` | Alignment tip specification |
| `root_link` / `root_axis` | `str` / `AxisIdentifier` | Alignment root specification |

`execute()`: calls `MoveJointsMotion` with PARK state plus optional `align`/`tip_link`/`tip_normal`/`root_link`/`root_normal` arguments forwarded to the giskardpy `JointPositionList` goal.

## FollowToolCenterPointPathAction

Moves the arm TCP along a trajectory of poses.

| Field | Type | Description |
|-------|------|-------------|
| `target_locations` | `PoseTrajectory` | Ordered sequence of TCP target poses |
| `arm` | `Arms` | Arm to move |

`execute()`: `MoveTCPWaypointsMotion(target_locations.poses, arm, allow_gripper_collision=True)`.

## Related

**Uses:** [[pycram.robot_plans.ActionDescription]], [[pycram.robot_plans.motions.robot_body]], [[pycram.robot_plans.motions.gripper]], [[pycram.view_manager.ViewManager]], [[sdt.robots.abstract_robot.AbstractRobot]], [[pycram.datastructures.Context]]

**Used by:** [[pycram.robot_plans.actions.composite]]

## Provenance

- `pycram/src/pycram/robot_plans/actions/core/robot_body.py` lines 1–241 (commit `0528d8cf3`) — all five action classes.
