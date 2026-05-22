---
id: pycram.view_manager.ViewManager
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/view_manager.py
    lines: [16, 71]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.robots.abstract_robot.AbstractRobot
used_by:
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.robot_body
  - pycram.robot_plans.actions.core.robot_body
  - pycram.robot_plans.motions.container
  - pycram.robot_plans.actions.core.container
status: stable
tags: [robot, arm, view, lookup, static]
last_ingest: 2026-05-18
---

_Static lookup table for robot sub-views: maps an `Arms` enum value + `AbstractRobot` annotation to the corresponding `KinematicChain` (arm), `Manipulator` (end-effector), or `Neck` (camera)._

## Purpose

`ViewManager` decouples motion designators from the concrete robot model. A motion designator does not hard-code which kinematic chain to use; instead it calls `ViewManager.get_end_effector_view(Arms.RIGHT, robot)` and receives the appropriate `Manipulator`. This makes plans robot-agnostic as long as the SDT robot annotation follows the `AbstractRobot` protocol.

The `@symbolic_function` decorator on `get_end_effector_view` makes it available as a krrood EQL symbolic predicate (used during underspecified action grounding).

## Key static methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_end_effector_view(arm, robot)` | `Manipulator \| None` | Returns the `Manipulator` for the given arm; symbolic-function decorated |
| `get_arm_view(arm, robot)` | `KinematicChain \| None` | Returns first matching arm chain |
| `get_all_arm_views(arm, robot)` | `Tuple[KinematicChain]` | Returns all matching chains (handles BOTH, single-arm robot) |
| `get_neck_view(robot)` | `Neck \| None` | Returns `robot.neck`; raises `ValueError` if absent |

## Arm resolution logic (`get_all_arm_views`)

- Single-arm robot: always returns `(robot.manipulator_chains[0],)` regardless of `arm` enum.
- `Arms.LEFT` → `(robot.left_arm,)`.
- `Arms.RIGHT` → `(robot.right_arm,)`.
- `Arms.BOTH` → `robot.arms` (tuple of all arm chains).
- Other → `None`.

## Related

**Uses:** [[sdt.robots.abstract_robot.AbstractRobot]]

**Used by:** [[pycram.robot_plans.motions.gripper]], [[pycram.robot_plans.motions.robot_body]], [[pycram.robot_plans.actions.core.robot_body]], [[pycram.robot_plans.motions.container]], [[pycram.robot_plans.actions.core.container]]

## Provenance

- `pycram/src/pycram/view_manager.py` lines 16–71 (commit `0528d8cf3`) — full `ViewManager` class.
