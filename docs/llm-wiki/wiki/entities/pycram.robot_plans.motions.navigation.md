---
id: pycram.robot_plans.motions.navigation
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/navigation.py
    lines: [1, 35]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - sdt.spatial_types.Pose
  - giskardpy.motion_statechart.tasks
used_by:
  - pycram.robot_plans.actions.core.NavigateAction
fields:
  target:
    type: sdt.spatial_types.Pose
    description: Goal pose for the robot base in the world frame.
  keep_joint_states:
    type: bool
    default: false
    description: If true, arm and head joints must not change during navigation.
status: stable
tags: [motion, navigation, cartesian, base]
last_ingest: 2026-05-19
---

_One `BaseMotion` subclass for driving the robot base: `MoveMotion` commands the robot root link to a target `Pose` via giskardpy `CartesianPose`._

## MoveMotion

Moves the robot's base (root link) to an absolute pose in the world frame.

| Field | Type | Description |
|-------|------|-------------|
| `target` | `Pose` | Goal pose for the robot base |
| `keep_joint_states` | `bool` | If `True`, arm/head joints must not change during navigation (default `False`) |

`_motion_chart` returns:
```python
CartesianPose(
    root_link=self.world.root,
    tip_link=self.robot.root,
    goal_pose=self.target,
)
```
This is a giskardpy Task that constrains the robot's root body to reach `target` in the world frame. The QP solver finds joint trajectories satisfying this constraint.

`perform()` is a no-op (inherited pattern from all motion designators — see [[pycram.robot_plans.motions.gripper]]).

## Related

**Uses:** [[pycram.robot_plans.BaseMotion]], [[sdt.spatial_types.Pose]]

**Used by:** [[pycram.robot_plans.actions.core.NavigateAction]]

**See also:** [[bridge.pycram-giskardpy]]

## Provenance

- `pycram/src/pycram/robot_plans/motions/navigation.py` lines 1–35 (commit `0528d8cf3`) — `MoveMotion` class definition.
