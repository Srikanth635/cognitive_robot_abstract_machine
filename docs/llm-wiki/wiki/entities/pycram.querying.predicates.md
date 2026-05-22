---
id: pycram.querying.predicates
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/querying/predicates.py
    lines: [1, 61]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - krrood.entity_query_language.predicate.Predicate
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.world_description.world_entity.Body
used_by:
  - pycram.robot_plans.actions.core.container
status: stable
tags: [pycram, predicate, gripper, occupancy, kinematic-structure, eql]
last_ingest: 2026-05-18
---

_Kinematic-structure-based gripper occupancy predicates: `GripperIsFree` and `GripperIsNotFree` check whether any body is attached under the manipulator's TCP in the world's kinematic tree._

## Purpose

`pycram.querying.predicates` provides `ActionDescription` pre-condition predicates that can be evaluated both concretely and as lazy EQL expressions (inheriting from `Predicate`). The occupancy check is **kinematic-structure-based**: a body must be *attached* to the kinematic tree under the tool-center-point (TCP) link to be counted, which corresponds to post-grasp attachment via `world.move_branch_with_fixed_connection`.

This is distinct from `sdt.reasoning.robot_predicates.is_body_in_gripper`, which uses ray-sampling between gripper finger meshes and detects objects that are *physically between the fingers* without requiring kinematic attachment.

## `GripperOccupancy` (base dataclass)

```python
@dataclass
class GripperOccupancy:
    manipulator: Manipulator
```

`check_man_occupancy(condition: Callable[[List[Body]], bool]) → bool`:

1. Calls `manipulator._world.get_kinematic_structure_entities_of_branch(manipulator.tool_frame)` — returns all KSE nodes in the sub-tree rooted at `tool_frame`.
2. Removes `tool_frame` itself from the list (the link is always present in its own branch).
3. Applies `condition(bodies_under_tcp)` and returns the result.

## `GripperIsFree(GripperOccupancy, Predicate)`

Returns `True` when no body is kinematically attached under the TCP:

```python
def __call__(self) -> bool:
    return self.check_man_occupancy(lambda bodies: len(bodies) == 0)
```

Used in `OpenAction.pre_condition` to guard that the gripper is empty before approaching the handle.

## `GripperIsNotFree(GripperOccupancy, Predicate)`

The inverse: returns `True` when at least one body is attached under the TCP:

```python
def __call__(self) -> bool:
    return self.check_man_occupancy(lambda bodies: len(bodies) != 0)
```

## Design note: kinematic attachment vs. physical contact

`GripperIsFree` checks kinematic attachment (bodies added to the world tree under the TCP via `PickUpAction`'s `move_branch_with_fixed_connection`). A body resting on the gripper fingers without being attached will NOT be detected. Prefer `sdt.reasoning.robot_predicates.is_body_in_gripper` or `bodies_in_gripper` for physical-contact checks.

## Related

**Uses:** [[krrood.entity_query_language.predicate.Predicate]], [[sdt.robots.abstract_robot.AbstractRobot]], [[sdt.world_description.world_entity.Body]]

**Used by:** [[pycram.robot_plans.actions.core.container]] (`OpenAction.pre_condition`)

**See also:** [[sdt.reasoning.robot_predicates]] — ray-sampling based gripper checks

## Provenance

- `pycram/src/pycram/querying/predicates.py:1-61` — `GripperOccupancy`, `GripperIsFree`, `GripperIsNotFree`.
