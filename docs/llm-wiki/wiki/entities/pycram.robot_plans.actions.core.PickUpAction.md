---
id: pycram.robot_plans.actions.core.PickUpAction
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/core/pick_up.py
    lines: [38, 264]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.motions.gripper
  - pycram.datastructures.grasp.GraspDescription
  - sdt.world_description.world_entity.Body
  - pycram.plans.factories
  - sdt.datastructures.GripperState
  - pycram.datastructures.enums.Arms
status: stable
tags: [action, pick-up, reach, grasp, manipulation]
last_ingest: 2026-05-19
used_by:
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.composite
  - pycram.robot_plans.actions.core.container
fields:
  object_designator:
    type: sdt.world_description.world_entity.Body
    description: The object to pick up.
  arm:
    type: pycram.datastructures.enums.Arms
    domain: [LEFT, RIGHT]
    description: Which arm to use.
  grasp_description:
    type: pycram.datastructures.grasp.GraspDescription
    description: Approach direction and grip configuration for the grasp.
---

_`PickUpAction`: two-phase grasp + lift. Also in this module: `ReachAction` (reach a target pose), `GraspingAction` (primitive reach-and-close without lift)._

## PickUpAction

Picks up a `Body` with a given arm and `GraspDescription`.

### Execution (two-phase)

```
Phase 1 — sequential subplan:
  MoveGripperMotion(OPEN, arm)
  ReachAction(target=object.global_pose, arm, grasp_description)
  MoveGripperMotion(CLOSE, arm)
  ↓ .perform()

  [world mutation — attach body to end-effector]:
  world.move_branch_with_fixed_connection(object_designator, end_effector.tool_frame)

Phase 2 — single-node subplan:
  MoveToolCenterPointMotion(lift_to_pose, arm,
      allow_gripper_collision=True, movement_type=TRANSLATION)
  ↓ .perform()
```

The world mutation between the two phases is non-obvious: it physically re-parents the
body in the SDT world model to the robot's tool frame **before** the lift motion runs.
This means the lift MSC moves both the arm and the attached object as a rigid unit.

### Pre / post conditions

- **Pre:** gripper is free (`GripperIsFree`) AND the grasp pose sequence is reachable
  (`pose_sequence_reachability_validator` with a deepcopy of the world).
- **Post:** gripper is not free OR body is in gripper frame (`is_body_in_gripper > 0.9`).

### Key fields

| Field               | Type               | Notes                                        |
| ------------------- | ------------------ | -------------------------------------------- |
| `object_designator` | `Body`             | The object to pick up.                       |
| `arm`               | `Arms`             | Which arm to use.                            |
| `grasp_description` | `GraspDescription` | Pose sequence and orientation for the grasp. |

---

## ReachAction

Reaches a target `Pose` using a given arm and `GraspDescription`. Used as a building
block by `PickUpAction` and `PlaceAction`.

```
sequential([
    MoveToolCenterPointMotion(pre_pose, arm, allow_gripper_collision=False),
    MoveToolCenterPointMotion(target_pose, arm, movement_type=CARTESIAN),
])
```

`grasp_description._pose_sequence(target_pose, object_designator, reverse)` computes
the pre-pose and final pose. `reverse=True` is used by `PlaceAction` to retract.

### Post-condition

End-effector is within 3 cm of target OR body is in gripper (`is_body_in_gripper > 0.9`).

---

## GraspingAction

Primitive grasp: moves to pre-pose, opens, moves to grasp pose, closes. No lift phase,
no world mutation. Used when only the grasp itself is needed (not pick-up-and-lift).

```
sequential([
    MoveToolCenterPointMotion(pre_pose, arm),
    MoveGripperMotion(OPEN, arm),
    MoveToolCenterPointMotion(grasp_pose, arm, allow_gripper_collision=True),
    MoveGripperMotion(CLOSE, arm, allow_gripper_collision=True),
])
```

---

## Related

- Base class: [[pycram.robot_plans.ActionDescription]]
- Motions used: [[pycram.robot_plans.motions.gripper]] (`MoveGripperMotion`,
  `MoveToolCenterPointMotion`)
- Grasp geometry: [[pycram.datastructures.grasp.GraspDescription]] (stub)
- Object type: [[sdt.world_description.world_entity.Body]]
- Called by (reverse): [[pycram.robot_plans.actions.core.PlaceAction]]
- Combinator: [[pycram.plans.factories]]

## Open questions

- `PickUpAction.execute` calls `grasp_description.grasp_pose_sequence(object)` but
  `ReachAction` calls `grasp_description._pose_sequence(target_pose, object, reverse)`.
  These are different methods — `grasp_pose_sequence` vs `_pose_sequence`. The public
  vs private distinction and different signatures suggest they return different things
  (one is object-centric, the other pose-centric). Clarify on GraspDescription ingest.
- The world mutation happens inside `execute()`, between two `.perform()` calls. This
  means the plan has already started executing when the world model changes. Whether
  this can cause issues during `re_perform()` / `replay()` is unclear.

## Provenance

- `pycram/src/pycram/robot_plans/actions/core/pick_up.py:38-264` at commit `0528d8cf3`.
