---
id: pycram.robot_plans.motions.container
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/container.py
    lines: [1, 59]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - sdt.world_description.world_entity.Body
  - pycram.view_manager.ViewManager
  - giskardpy.motion_statechart.graph_node.Goal
  - giskardpy.motion_statechart.goals
  - pycram.datastructures.enums.Arms
used_by:
  - pycram.robot_plans.actions.core.container
fields:
  OpeningMotion:
    object_part:
      type: sdt.world_description.world_entity.Body
      description: SDT Body representing the container handle or knob to open.
    arm:
      type: pycram.datastructures.enums.Arms
      description: Which arm to use; tool frame resolved via ViewManager.
  ClosingMotion:
    object_part:
      type: sdt.world_description.world_entity.Body
      description: SDT Body representing the container handle or knob to close.
    arm:
      type: pycram.datastructures.enums.Arms
      description: Which arm to use; tool frame resolved via ViewManager.
status: stable
tags: [motion, container, open, close, giskardpy]
last_ingest: 2026-05-19
---

_Two `BaseMotion` subclasses for articulated-container control: `OpeningMotion` and `ClosingMotion`. Both delegate joint trajectory generation to giskardpy `Open`/`Close` goals using the arm's tool frame as the tip link._

## OpeningMotion

Drives a container joint (drawer/door handle) to its open position.

| Field | Type | Description |
|-------|------|-------------|
| `object_part` | `Body` | SDT `Body` representing the handle/knob |
| `arm` | `Arms` | Which arm to use |

`_motion_chart`:
```python
tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
return Open(tip_link=tip, environment_link=self.object_part)
```
The giskardpy `Open` goal moves the joint until it reaches its maximum position.

## ClosingMotion

Drives a container joint to a near-closed position (`goal_joint_state=0.01`).

| Field | Type | Description |
|-------|------|-------------|
| `object_part` | `Body` | SDT `Body` representing the handle/knob |
| `arm` | `Arms` | Which arm to use |

`_motion_chart`:
```python
tip = ViewManager().get_end_effector_view(self.arm, self.robot).tool_frame
return Close(tip_link=tip, environment_link=self.object_part, goal_joint_state=0.01)
```
`goal_joint_state=0.01` leaves a 1 cm gap — prevents the gripper from being pinched by a fully-closed joint.

## Related

**Uses:** [[pycram.robot_plans.BaseMotion]], [[sdt.world_description.world_entity.Body]], [[pycram.view_manager.ViewManager]]

**Used by:** [[pycram.robot_plans.actions.core.container]]

## Provenance

- `pycram/src/pycram/robot_plans/motions/container.py` lines 1–59 (commit `0528d8cf3`) — `OpeningMotion` and `ClosingMotion`.
