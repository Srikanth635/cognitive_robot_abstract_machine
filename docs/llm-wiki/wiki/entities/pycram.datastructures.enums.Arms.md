---
id: pycram.datastructures.enums.Arms
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/datastructures/enums.py
    lines: [63, 72]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
values:
  - LEFT
  - RIGHT
  - BOTH
uses: []
used_by:
  - pycram.datastructures.grasp.GraspDescription
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.core.container
  - pycram.robot_plans.actions.core.robot_body
  - pycram.robot_plans.actions.composite
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.container
status: stable
tags: [pycram, enum, arms, manipulator, gripper]
last_ingest: 2026-05-19
---

_Enum: selects which robot arm(s) to use for a manipulation action._

| Value | Int | Meaning |
|---|---|---|
| `LEFT` | 0 | Left arm / left gripper |
| `RIGHT` | 1 | Right arm / right gripper |
| `BOTH` | 2 | Both arms (e.g. bimanual tasks) |

`Arms` is an `IntEnum`. Used as the `arm` parameter in `PickUpAction`, `PlaceAction`, and `MoveGripperMotion`, and as the `arm` field on `GraspPose`.

## Related

- **Used by:** [[pycram.datastructures.grasp.GraspDescription]] (`GraspPose.arm`), [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.actions.core.PlaceAction]], [[pycram.robot_plans.motions.gripper]]

## Provenance

- `pycram/src/pycram/datastructures/enums.py:63-72` — `Arms(IntEnum)` class.
