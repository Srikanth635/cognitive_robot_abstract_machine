---
id: pycram.robot_plans.motions.robot_body
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/robot_body.py
    lines: [1, 91]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - sdt.spatial_types.Pose
  - sdt.robots.abstract_robot.AbstractRobot
  - pycram.view_manager.ViewManager
  - giskardpy.motion_statechart.tasks
used_by:
  - pycram.robot_plans.actions.core.robot_body
fields:
  MoveJointsMotion:
    names:
      type: str
      description: List of joint (connection) names to move; resolved to SDT Connection objects via world.get_connection_by_name.
    positions:
      type: float
      description: Target position for each named joint (parallel list to names).
  LookingMotion:
    target:
      type: sdt.spatial_types.Pose
      description: World-frame point to look at.
    camera:
      type: sdt.robots.abstract_robot.AbstractRobot
      description: SDT Camera annotation from AbstractRobot; provides root link and forward-facing axis for the giskardpy Pointing goal.
status: stable
tags: [motion, joints, looking, torso, camera, giskardpy]
last_ingest: 2026-05-19
---

_Two `BaseMotion` subclasses for whole-body joint and camera control: `MoveJointsMotion` commands named joints to target positions; `LookingMotion` points the robot's camera at a target pose._

## MoveJointsMotion

Commands a set of named robot joints to specific positions via giskardpy `JointPositionList`.

| Field | Type | Description |
|-------|------|-------------|
| `names` | `List[str]` | Joint (connection) names to move |
| `positions` | `List[float]` | Target position for each named joint |
| `align` | `bool` | Optional: align end-effector axis during motion |
| `tip_link` / `tip_normal` | `str` / `Vector3` | Axis alignment tip parameters |
| `root_link` / `root_normal` | `str` / `Vector3` | Axis alignment root parameters |

`_motion_chart`:
```python
dofs = [self.world.get_connection_by_name(name) for name in self.names]
return JointPositionList(
    goal_state=JointState.from_mapping(dict(zip(dofs, self.positions)))
)
```
Names are resolved to SDT `Connection` objects via `world.get_connection_by_name`. `JointPositionList` and `JointState` are giskardpy types.

Used by `ParkArmsAction`, `MoveTorsoAction`, `CarryAction` — they all translate high-level robot states (PARK, TorsoState) into `(names, positions)` pairs from `StaticJointState`.

## LookingMotion

Points the robot camera at a 3D goal point using giskardpy `Pointing`.

| Field | Type | Description |
|-------|------|-------------|
| `target` | `Pose` | World-frame point to look at |
| `camera` | `Camera` | SDT `Camera` annotation from `AbstractRobot` |

`_motion_chart`:
```python
self.camera.forward_facing_axis.reference_frame = self.camera.root
return Pointing(
    root_link=self.robot.torso.root,
    tip_link=self.camera.root,
    goal_point=self.target.to_position(),
    pointing_axis=self.camera.forward_facing_axis,
)
```
Mutates `camera.forward_facing_axis.reference_frame` in-place before creating the goal — this is a side effect on the robot annotation. Whether it's safe to call concurrently is unknown.

## Related

**Uses:** [[pycram.robot_plans.BaseMotion]], [[sdt.spatial_types.Pose]], [[sdt.robots.abstract_robot.AbstractRobot]]

**Used by:** [[pycram.robot_plans.actions.core.robot_body]]

## Open questions

- `LookingMotion._motion_chart` mutates `camera.forward_facing_axis.reference_frame` as a side effect. This is set every time `_motion_chart` is accessed (it's a `@property`). Concurrent access from two plan threads on the same robot annotation could race.

## Provenance

- `pycram/src/pycram/robot_plans/motions/robot_body.py` lines 1–91 (commit `0528d8cf3`) — `MoveJointsMotion`, `LookingMotion`.
