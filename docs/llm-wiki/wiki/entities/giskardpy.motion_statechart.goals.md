---
id: giskardpy.motion_statechart.goals
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/goals/cartesian_goals.py
    lines: [1, 149]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/goals/open_close.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/goals/templates.py
    lines: [1, 65]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/goals/collision_avoidance.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.Goal
  - giskardpy.motion_statechart.graph_node.CancelMotion
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
  - giskardpy.motion_statechart.tasks
used_by:
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.container
status: stable
tags: [giskardpy, goal, cartesian, navigation, open-close, collision-avoidance, sequence, parallel]
last_ingest: 2026-05-17
---

_Bundled page: all concrete `Goal` subclasses — navigation, cartesian, open/close, and collision avoidance._

## Overview

Concrete `Goal` subclasses live under `goals/`. Each overrides `expand(context)` to instantiate child `Task` (and optionally child `Goal`) nodes and wire them together. Goals never add QP constraints directly — they compose `Task` children. `build()` optionally produces an observation expression by ANDing children's `observation_variable`.

## Structural templates (`templates.py`)

Documented in [[giskardpy.motion_statechart.graph_node.Goal]]:
- `Sequence(Goal)` — children start in order, own observation = last child's.
- `Parallel(Goal)` — all children start together, own observation = `minimum_success ≤ count(TRUE children)`.

## Navigation goals (`cartesian_goals.py`)

### `DifferentialDriveBaseGoal(Sequence)`

Three-step sequence for differential drive navigation:

```
step1: CartesianOrientation  — face the goal position
step2: CartesianPose         — drive to goal (with intermediate facing orientation)
step3: CartesianPose         — final orientation adjustment at goal position
```

`expand()` auto-discovers the single `DifferentialDrive` connection from the world if `diff_drive_connection=None`; raises `NodeInitializationError` if zero or >1 found. Goal orientation for step1 is computed from the vector from current to goal position cross `Vector3.Z`.

### `CartesianPoseStraight(Parallel)`

Parallel composition of `CartesianPositionStraight + CartesianOrientation`. Constrains the tip link to approach the goal via a straight-line path in Cartesian space.

## Articulated object goals (`open_close.py`)

### `Open(Goal)`

Moves a robot's `tip_link` to open a container with one active DOF (drawer, door). `expand()` walks ancestors of `environment_link` via `get_first_parent_connection_of_type(ActiveConnection1DOF)` to find the joint. Adds two children in parallel:
- `JointPositionList` — moves the container joint to `goal_joint_state` (default: `upper.position`)
- `CartesianPose` — keeps the hand rigidly fixed to the handle

`build()` returns observation = AND of all children's `observation_variable`.

### `Close(Open)`

Mirror of `Open` with `goal_joint_state` defaulting to `lower.position` instead.

## Collision avoidance goals (`collision_avoidance.py`)

Contains `_ExternalCollisionAvoidanceNode` and related tasks that add repulsive QP constraints based on `ExternalCollisionVariableManager` / `SelfCollisionVariableManager` distance queries. These are internal helpers — the main entry point is the `Goal` wrapper that sets up the correct `CancelMotion` trigger with `CollisionViolatedError` when hard limits are breached.

## `GoalBindingPolicy`

Controls **when** a cartesian goal pose is frozen into a constant:
- `Bind_at_build` — snapshot the goal FK expression at compile time
- `Bind_on_start` — snapshot at the moment the node enters RUNNING (default for `CartesianTask`)

Choosing `Bind_on_start` is important for goals in moving reference frames where the target should be resolved relative to where the robot/object is at start time, not at plan-compilation time.

## Related

- **Uses:** [[giskardpy.motion_statechart.graph_node.Goal]], [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]], [[giskardpy.motion_statechart.tasks]]
- **Used by:** pycram motion designators (`_motion_chart` returns a `Goal` subclass)

## Provenance

- `goals/cartesian_goals.py:1-149` — `DifferentialDriveBaseGoal`, `CartesianPoseStraight`.
- `goals/open_close.py:1-80` — `Open`, `Close`.
- `goals/collision_avoidance.py:1-80` — `_CollisionAvoidanceTask`, `_ExternalCollisionAvoidanceNode` internals.
- `goals/templates.py:1-65` — `Sequence`, `Parallel` (also documented in [[giskardpy.motion_statechart.graph_node.Goal]]).
