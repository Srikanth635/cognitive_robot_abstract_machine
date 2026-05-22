---
id: sdt.datastructures.GripperState
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/datastructures/definitions.py
    lines: [7, 10]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - bridge.pycram-sdt
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.core.robot_body
  - pycram.robot_plans.actions.core.container
status: stable
tags: [sdt, gripper, state, enum, joint-state-type]
last_ingest: 2026-05-17
---

_`JointStateType` enum with three values — `OPEN`, `CLOSE`, `MEDIUM` — used as the `state_type` tag for gripper `JointState` instances and as a gripper command in pycram motion designators._

`GripperState` subclasses `JointStateType` (also an `Enum`). The enum value is passed as the `gripper_state` parameter to `MoveGripperMotion`, which routes it to the giskardpy gripper goal.

**Note:** `sdt.datastructures.joint_state` defines a separate type alias `GripperState = JointState` for the target configuration object itself. These are two distinct uses of the name — the enum here is the command selector, not the configuration container.

## Related

- **Used by:** [[pycram.robot_plans.motions.gripper]], [[pycram.robot_plans.actions.core.PickUpAction]], [[pycram.robot_plans.actions.core.PlaceAction]]

## Provenance

- `definitions.py:7-10` — `GripperState(JointStateType)` enum.
