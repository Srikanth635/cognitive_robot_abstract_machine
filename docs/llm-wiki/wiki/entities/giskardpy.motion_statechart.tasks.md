---
id: giskardpy.motion_statechart.tasks
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/tasks/cartesian_tasks.py
    lines: [1, 220]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/tasks/joint_tasks.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.Task
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
used_by:
  - giskardpy.motion_statechart.goals
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.navigation
  - pycram.robot_plans.motions.robot_body
status: stable
tags: [giskardpy, task, cartesian, joint, constraint, qp, leaf-node]
last_ingest: 2026-05-17
---

_Bundled page: all concrete MSC leaf `Task` subclasses — cartesian and joint-space QP constraint generators._

## Overview

`Task` subclasses are the **leaf nodes** of the MSC graph. They override `build(context)` to populate `NodeArtifacts.constraints` with QP equality/inequality constraints and set `NodeArtifacts.observation` to a symbolic boolean signalling goal achievement. They do not add children.

## `CartesianTask(Task, ABC)` — base class

Adds goal-binding policy support. Subclasses declare `goal_reference_frame` abstractly. At `build()`, a `ForwardKinematicsBinding` is created for `root_link → goal_reference_frame`; it resolves `root_T_goal_reference_frame` either at compile time (`Bind_at_build`) or at RUNNING entry (`Bind_on_start`, default).

### Cartesian task inventory

| Class | Controls | Observation |
|---|---|---|
| `CartesianPosition` | x/y/z of tip_link | euclidean distance < `threshold` (0.01 m) |
| `CartesianPositionStraight` | x/y/z constrained to straight-line path | euclidean distance < threshold |
| `CartesianPositionTrajectory` | tip follows dense waypoint list with look-ahead | reaches final waypoint |
| `CartesianOrientation` | rotation of tip_link | rotational error < threshold |
| `CartesianPose` | position + orientation (parallel composition internally) | both position AND orientation |

`CartesianPosition.default_reference_velocity = 0.2 m/s`. The `reference_velocity` parameter normalises the QP equality constraint (scales how aggressively the controller approaches the goal).

`CartesianPositionTrajectory` maintains a `current_index` into the `goal_points` list, advancing via a look-ahead distance mechanism. It requires all points share the same `reference_frame`.

## Joint task

### `JointPositionList(Task)`

Moves a set of joints to a `JointState` target simultaneously.

```python
JointPositionList(
    goal_state=JointState({connection: target_pos, ...}),
    threshold=0.01,                       # observation fires when all errors < threshold
    weight=DefaultWeights.WEIGHT_BELOW_CA,
    max_velocity=1.0,
)
```

For each `(connection, target)` pair it adds one equality constraint. Continuous revolute joints (`RevoluteConnection` without position limits) use `shortest_angular_distance` for wrapping. Observation = `logic_all(errors < threshold)`. Raises `NodeInitializationError` if `goal_state` is empty.

## `DefaultWeights` reference

| Constant | Value | Usage |
|---|---|---|
| `WEIGHT_MAX` | 10000 | Hard constraints (safety stops) |
| `WEIGHT_ABOVE_CA` | 2500 | Primary motion goals (above collision avoidance) |
| `WEIGHT_COLLISION_AVOIDANCE` | 50 | Collision avoidance repulsive tasks |
| `WEIGHT_BELOW_CA` | 1 | Secondary joint goals (below CA priority) |

## Related

- **Uses:** [[giskardpy.motion_statechart.graph_node.Task]], [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- **Used by:** [[giskardpy.motion_statechart.goals]] (Goals compose Tasks via `expand()`)

## Provenance

- `tasks/cartesian_tasks.py:1-220` — `CartesianTask`, `CartesianPosition`, `CartesianPositionStraight`, `CartesianPositionTrajectory`, `CartesianOrientation`, `CartesianPose`.
- `tasks/joint_tasks.py:1-80` — `JointPositionList`.
