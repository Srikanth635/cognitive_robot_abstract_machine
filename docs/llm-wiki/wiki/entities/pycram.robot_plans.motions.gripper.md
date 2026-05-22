---
id: pycram.robot_plans.motions.gripper
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/gripper.py
    lines: [1, 215]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - sdt.spatial_types.Pose
  - sdt.world_description.world_entity.Body
  - pycram.datastructures.grasp.GraspDescription
  - pycram.view_manager.ViewManager
  - sdt.datastructures.GripperState
  - giskardpy.motion_statechart.graph_node.Goal
  - giskardpy.motion_statechart.tasks
  - giskardpy.motion_statechart.goals
  - pycram.datastructures.enums.Arms
used_by:
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.core.NavigateAction
  - pycram.robot_plans.actions.core.container
  - pycram.robot_plans.actions.core.robot_body
fields:
  MoveGripperMotion:
    motion:
      type: sdt.datastructures.GripperState
      domain: [OPEN, CLOSE]
      description: Open or close the gripper.
    gripper:
      type: pycram.datastructures.enums.Arms
      description: Which arm's gripper to move.
    allow_gripper_collision:
      type: bool
      description: If true, permits contact during gripper motion (passed to MSC); optional.
  MoveToolCenterPointMotion:
    target:
      type: sdt.spatial_types.Pose
      description: Goal pose for the TCP.
    arm:
      type: pycram.datastructures.enums.Arms
      description: Which arm's TCP to move.
    movement_type:
      type: str
      domain: [CARTESIAN, TRANSLATION]
      default: CARTESIAN
      description: CARTESIAN constrains position and orientation; TRANSLATION constrains position only.
  ReachMotion:
    object_designator:
      type: sdt.world_description.world_entity.Body
      description: Object to approach.
    arm:
      type: pycram.datastructures.enums.Arms
      description: Which arm to use.
    grasp_description:
      type: pycram.datastructures.grasp.GraspDescription
      description: Used to compute pre-pose and target pose for the approach sequence.
    reverse_pose_sequence:
      type: bool
      default: false
      description: If true, reverses the approach order (retreat path used by PlaceAction).
  MoveTCPWaypointsMotion:
    waypoints:
      type: sdt.spatial_types.Pose
      description: Ordered list of target TCP poses.
    arm:
      type: pycram.datastructures.enums.Arms
      description: Which arm's TCP to move through the waypoints.
status: stable
tags: [motion, gripper, tcp, cartesian, giskardpy]
last_ingest: 2026-05-19
---

_Four `BaseMotion` subclasses for arm/gripper control: `MoveGripperMotion`, `MoveToolCenterPointMotion`, `ReachMotion`, `MoveTCPWaypointsMotion`. All are leaf nodes; none perform directly — they produce giskardpy goals via `_motion_chart`._

## Key pattern: perform() = pass

All four motions override `perform()` as a no-op (`return`). The actual robot movement
is entirely driven by `_motion_chart` → giskardpy `Task` → `MotionExecutor` MSC, as
described in [[pycram.plans.MotionNode]]. A `MotionNode._perform()` call to
`self.motion.perform()` does nothing; the parent `ActionNode` harvests
`motion.motion_chart` when constructing the MSC.

## MoveGripperMotion

Opens or closes the gripper via a `JointPositionList` goal.

| Field | Type | Notes |
|---|---|---|
| `motion` | `GripperState` | `OPEN` or `CLOSE`. |
| `gripper` | `Arms` | Which arm's gripper. |
| `allow_gripper_collision` | `Optional[bool]` | Passed to MSC; permits contact during close. |

`_motion_chart`: `JointPositionList(goal_state=arm.get_joint_state_by_type(motion), name="OpenGripper"|"CloseGripper")`

## MoveToolCenterPointMotion

Moves the Tool Center Point (TCP) of a specific arm to a target `Pose`.

| Field | Type | Notes |
|---|---|---|
| `target` | `Pose` | Goal pose. |
| `arm` | `Arms` | Which arm's TCP. |
| `allow_gripper_collision` | `Optional[bool]` | — |
| `movement_type` | `MovementType` | `CARTESIAN` (default) or `TRANSLATION`. |

`_motion_chart` root link selection: `self.world.root` if `robot.full_body_controlled`,
else `self.robot.root`. This makes mobile manipulators plan in world frame while
fixed-base robots plan in their own root frame.

- `CARTESIAN` → `CartesianPose(root, tip, goal_pose)`
- `TRANSLATION` → `CartesianPosition(root, tip, goal_point)` (position only, no orientation)

## ReachMotion

Builds a `Sequence` of `CartesianPose` goals for approaching an object.

| Field | Type | Notes |
|---|---|---|
| `object_designator` | `Body` | Object to approach. |
| `arm` | `Arms` | — |
| `grasp_description` | `GraspDescription` | Used to compute pre-pose and target pose. |
| `movement_type` | `MovementType` | Default `CARTESIAN`. |
| `reverse_pose_sequence` | `bool` | If `True`, reverses the approach (used for retreat). |

`_motion_chart`: `Sequence(nodes=[CartesianPose(pre_pose), CartesianPose(target_pose)])`.

Note: `ReachMotion` exists alongside `ReachAction` (in `pick_up.py`). `ReachAction` uses
`MoveToolCenterPointMotion` internally; `ReachMotion` is the lower-level `BaseMotion`
alternative that constructs the pose sequence itself.

## MoveTCPWaypointsMotion

Moves the TCP through an ordered list of waypoints, each as a `CartesianPose` in a
giskardpy `Sequence`.

| Field | Type | Notes |
|---|---|---|
| `waypoints` | `List[Pose]` | Ordered list of target poses. |
| `arm` | `Arms` | — |
| `movement_type` | `WaypointsMovementType` | Default `ENFORCE_ORIENTATION_FINAL_POINT`. |

## Related

- Base class: [[pycram.robot_plans.BaseMotion]]
- Consumed by action nodes: [[pycram.robot_plans.actions.core.PickUpAction]],
  [[pycram.robot_plans.actions.core.PlaceAction]]
- Pose type: [[sdt.spatial_types.Pose]]
- Object type: [[sdt.world_description.world_entity.Body]]
- Grasp geometry: [[pycram.datastructures.grasp.GraspDescription]] (stub)
- View lookup: [[pycram.view_manager.ViewManager]] (stub)

## Open questions

- `ReachMotion` and `ReachAction` (in `pick_up.py`) are parallel implementations of
  the same concept. `ReachAction` is preferred in pick-up/place flows. When is
  `ReachMotion` the right choice? Likely when no subplan construction is needed
  (i.e., when called from a lower-level MSC directly). Needs clarification.
- The `TODO` comment in `ReachMotion._calculate_pose_sequence` (`# TODO: Maybe put
  these values in the semantic annotates`) indicates hard-coded 0.05 m pre-approach
  offset — likely should come from the semantic description of the manipulator.

## Provenance

- `pycram/src/pycram/robot_plans/motions/gripper.py:1-215` at commit `0528d8cf3`.
