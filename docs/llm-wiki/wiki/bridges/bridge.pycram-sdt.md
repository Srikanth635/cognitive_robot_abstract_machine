---
id: bridge.pycram-sdt
kind: bridge
package: cross
source_paths:
  - path: pycram/src/pycram/plans/failures.py
    lines: [1, 20]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/actions/core/pick_up.py
    lines: [1, 264]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/motions/gripper.py
    lines: [1, 215]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - concept.semantic-annotation
  - pycram.datastructures.Context
  - sdt.world.World
  - sdt.world_description.connections
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.spatial_types.Pose
  - sdt.world_description.world_entity.Body
  - sdt.datastructures.GripperState
used_by: []
status: stable
tags: [bridge, pycram, sdt, world, robot, coupling]
last_ingest: 2026-05-17
---

_Runtime interface between pycram and semantic_digital_twin: plan context, world manipulation, and symbolic failure types._

## Purpose

pycram consumes SDT via three channels:

1. **Context injection** — `pycram.datastructures.Context` carries `world: sdt.World` and
   `robot: sdt.AbstractRobot`. Every `PlanNode` and `Designator` property delegates through
   `plan_node → plan → context` to reach the live SDT world and robot.
2. **World mutation** — `PickUpAction` and `PlaceAction` directly call
   `world.move_branch_with_fixed_connection` and `Connection6DoF.create_with_dofs` to
   attach/detach the grasped body in the kinematic tree. This happens **inside**
   `ActionDescription.execute()`, interleaved with motion execution.
3. **Symbolic failure types** — `pycram.plans.failures` imports SDT types at **runtime** (not
   `TYPE_CHECKING`): `sdt.spatial_types.Pose` (in `NavigationGoalNotReachedError`) and
   `sdt.world_description.world_entity.Body` (in `BodyUnfetchable`). This is the first hard
   compile-time coupling between the packages.

## Coupling inventory

| Site | pycram side | SDT side | Kind |
|---|---|---|---|
| `Context.world` | `pycram.datastructures.Context` | `sdt.world.World` | runtime injection |
| `Context.robot` | `pycram.datastructures.Context` | `sdt.robots.abstract_robot.AbstractRobot` | runtime injection |
| `PickUpAction.execute` | `pycram.robot_plans.actions.core.PickUpAction` | `sdt.world.World.move_branch_with_fixed_connection` | direct call |
| `PlaceAction.execute` | `pycram.robot_plans.actions.core.PlaceAction` | `sdt.world.World` + `Connection6DoF` | direct call |
| `MoveToolCenterPointMotion._motion_chart` | `pycram.robot_plans.motions.gripper` | `AbstractRobot.full_body_controlled`, `World.root` | property access |
| `MoveGripperMotion._motion_chart` | `pycram.robot_plans.motions.gripper` | `sdt.datastructures.GripperState` | enum lookup |
| `NavigationGoalNotReachedError` | `pycram.plans.failures` | `sdt.spatial_types.Pose` | runtime import |
| `BodyUnfetchable` | `pycram.plans.failures` | `sdt.world_description.world_entity.Body` | runtime import |

## Key observations

- The world mutation inside `PickUpAction.execute()` calls `world.move_branch_with_fixed_connection`,
  which is decorated `@atomic_world_modification` in SDT — so atomicity and locking are guaranteed
  by SDT's own decorator, not by the pycram caller.
- The `failures.py` runtime imports mean that importing `pycram.plans.failures` at module load
  time triggers SDT imports. Any test or tool that wants a pycram-only environment must patch or
  mock these imports.

## Related

- Context carrier: [[pycram.datastructures.Context]]
- SDT world: [[sdt.world.World]]
- SDT robot: [[sdt.robots.abstract_robot.AbstractRobot]]
- Pick-up coupling: [[pycram.robot_plans.actions.core.PickUpAction]]
- Failure coupling: [[pycram.plans.failures.PlanFailure]]
- Symmetric bridge: [[bridge.pycram-giskardpy]]

## Open questions

- Are the runtime SDT imports in `failures.py` intentional (coupling accepted) or unintentional
  (should be `TYPE_CHECKING`-only)? The imports are not guarded, so the coupling is real.
- Concrete `AbstractRobot` subclasses (robot-model specific) not yet ingested — Phase 6 target.

## Provenance

- `pycram/src/pycram/plans/failures.py:1-20` at commit `0528d8cf3` — runtime SDT import lines.
- `pycram/src/pycram/robot_plans/actions/core/pick_up.py:1-264` at commit `0528d8cf3` — PickUpAction world mutation.
- `pycram/src/pycram/robot_plans/motions/gripper.py:1-215` at commit `0528d8cf3` — gripper motions accessing `robot.full_body_controlled`.
