---
id: pycram.robot_plans.actions.core.PlaceAction
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/core/placing.py
    lines: [35, 134]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.motions.gripper
  - pycram.datastructures.grasp.GraspDescription
  - sdt.world_description.world_entity.Body
  - sdt.spatial_types.Pose
  - pycram.plans.factories
  - sdt.datastructures.GripperState
  - pycram.datastructures.enums.Arms
used_by:
  - pycram.robot_plans.actions.composite
fields:
  object_designator:
    type: sdt.world_description.world_entity.Body
    description: The body being placed.
  target_location:
    type: sdt.spatial_types.Pose
    description: World pose where the body should be placed.
  arm:
    type: pycram.datastructures.enums.Arms
    domain: [LEFT, RIGHT]
    description: The arm currently holding the object.
  grasp_description:
    type: pycram.datastructures.grasp.GraspDescription
    derived_from: plan_history.PickUpAction.grasp_description
    description: Grasp configuration; recovered automatically from the most recent PickUpAction in the plan history. Falls back to a default front-approach GraspDescription if no prior pick-up is found.
status: stable
tags: [action, place, manipulation, world-mutation]
last_ingest: 2026-05-19
---

_Places a held `Body` at a target `Pose`. Reverses the grasp approach, opens the gripper, detaches the body from the robot by rewriting its parent connection in the SDT world model, then retracts._

## Purpose

`PlaceAction` is the inverse of `PickUpAction`. It:

1. **Looks up the prior pick-up** from the plan history to reuse its `GraspDescription`.
2. **Reaches the target location in reverse** (approach from target → pre-pose).
3. **Opens the gripper**.
4. **Detaches the body** from the robot arm in the SDT world model.
5. **Retracts** the arm.

## Execution flow

```
1. Recover grasp: plan_node.get_previous_node_by_designator_type(PickUpAction)
   → uses previous_pick.designator.grasp_description
     (falls back to default GraspDescription if no prior pick-up found)

2. sequential([
     ReachAction(target_location, arm, previous_grasp, object,
                 reverse_reach_order=True),
     MoveGripperMotion(GripperState.OPEN, arm),
   ]).perform()

3. [world mutation — detach body]:
   world.remove_connection(object.parent_connection)
   connection = Connection6DoF.create_with_dofs(parent=world_root, child=object)
   world.add_connection(connection)
   connection.origin = compute_forward_kinematics(world_root, object)

4. execute_single(MoveToolCenterPointMotion(retract_pose, arm)).perform()
```

## Key design points

**History lookup**: `get_previous_node_by_designator_type(PickUpAction)` traverses the
plan backward to find the most recent `ActionNode` whose designator is a `PickUpAction`.
This gives the grasp description without having to pass it explicitly as a parameter.
If no prior pick-up exists in the plan, `PlaceAction` creates a default front-approach
`GraspDescription` — meaning it can be used standalone (though the grasp direction
may be wrong).

**World detachment**: After opening the gripper, the code removes the body's current
parent connection (which links it to the end-effector) and creates a `Connection6DoF`
parenting it to the world root at its current global pose. This is the SDT equivalent
of "releasing" the object.

## Key fields

| Field | Type | Notes |
|---|---|---|
| `object_designator` | `Body` | The body being placed. |
| `target_location` | `Pose` | World pose where the body should land. |
| `arm` | `Arms` | The arm currently holding the object. |

## Pre / post conditions

- **Pre:** body is in gripper (`is_body_in_gripper > 0.9`) OR gripper is not free.
- **Post:** gripper is free AND body is not in gripper AND body is within 3 cm of target.

## Related

- Base class: [[pycram.robot_plans.ActionDescription]]
- Depends on prior: [[pycram.robot_plans.actions.core.PickUpAction]] (history lookup)
- Motions: [[pycram.robot_plans.motions.gripper]]
- Object type: [[sdt.world_description.world_entity.Body]]
- Target type: [[sdt.spatial_types.Pose]]
- Combinator: [[pycram.plans.factories]]

## Open questions

- The forward-kinematics call `compute_forward_kinematics(world_root, object)` is
  used to preserve the object's current global pose when reparenting. If the gripper
  hasn't fully opened yet when this runs, the pose may be slightly off. Worth verifying
  execution order during Phase 5 (SDT world-mutation semantics).
- `Connection6DoF.create_with_dofs` is a new SDT entity not yet in the wiki — stub
  candidates for Phase 5/6.

## Provenance

- `pycram/src/pycram/robot_plans/actions/core/placing.py:35-134` at commit `0528d8cf3`.
