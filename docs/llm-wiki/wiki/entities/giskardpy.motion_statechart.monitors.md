---
id: giskardpy.motion_statechart.monitors
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/monitors/monitors.py
    lines: [1, 81]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/monitors/payload_monitors.py
    lines: [1, 107]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/monitors/cartesian_monitors.py
    lines: [1, 232]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/monitors/joint_monitors.py
    lines: [1, 41]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
used_by:
  - concept.motion-statechart
status: stable
tags: [giskardpy, monitor, trinary, cartesian, joint, convergence, payload]
last_ingest: 2026-05-17
---

_Bundled page: all MSC monitor node families — convergence, payload, cartesian geometry, joint position, and timing monitors._

## Overview

Monitors are `MotionStatechartNode` subclasses that produce a trinary observation state but add **no QP constraints**. They wire `EndMotion` / `CancelMotion` start conditions or serve as guards in `Sequence` / `Parallel` templates. All monitors return their observation purely from `NodeArtifacts.observation` (symbolic) or `on_tick()` (imperative).

## `LocalMinimumReached` — convergence monitor

Fires TRUE when all active DOFs have been below their velocity threshold for >1 second of trajectory time.

```python
LocalMinimumReached(
    joint_convergence_threshold=0.01,  # fraction of max DOF velocity
    minimum_threshold=0.01,            # clamp floor for per-DOF threshold
    maximum_threshold=0.06,            # clamp ceiling
    windows_size=1,
)
```

The per-DOF threshold is `clamp(dof.limits.upper.velocity * joint_convergence_threshold, min, max)`. The `traj_longer_than_1_sec` guard prevents spurious TRUE at the very start of a motion.

## Payload monitors (`payload_monitors.py`)

Timing/counting monitors that use `on_tick()` imperative logic:

| Class | Fires TRUE when |
|---|---|
| `CheckControlCycleCount` | cumulative control cycles > `threshold` |
| `CountControlCycles` | `control_cycles` ticks elapsed since RUNNING start |
| `CountSeconds` | `seconds` wall-clock seconds elapsed since RUNNING start |
| `Pulse` | first `length` ticks after RUNNING start; FALSE thereafter |
| `Print` | always TRUE; prints `message` as a side-effect each tick |

`CountSeconds` uses `time.monotonic` (injectable via `_now` for testing). `CountControlCycles` and `Pulse` both reset their counter in `on_start`.

## `ThreadedPayloadMonitor` — async observation

Abstract base in `monitors.py`. Subclasses implement `__call__()` which runs in a daemon thread when `start_condition` becomes TRUE. The result is written to `self.state: ObservationStateValues`; the node's `on_tick()` returns the cached value. Useful for heavy observations (IK reachability, collision queries) that can lag the control loop.

## Cartesian monitors (`cartesian_monitors.py`)

All use FK expressions from `world._forward_kinematic_manager.compose_expression(root, tip)`. Their `observation_expression` is a symbolic CasADi boolean.

| Class | Condition |
|---|---|
| `PoseReached` | position AND orientation errors below separate thresholds |
| `PositionReached` | euclidean distance from `goal_point` < `threshold` |
| `OrientationReached` | rotational error < `threshold` |
| `PointingAt` | axis-to-goal-point line distance < `threshold` |
| `VectorsAligned` | angle between tip and goal vectors < `threshold` |
| `DistanceToLine` | distance from tip to a line segment < `threshold` |
| `InWorldSpace` | diff-drive robot's tip is within `xyz` tolerances of the drive link (projects to floor) |

`PoseReached` and related monitors accept an `absolute` flag. When `False` (default), the goal is frozen at the moment the node starts (`update_expression_on_starting`), allowing goals expressed in moving frames to be resolved once and tracked thereafter.

## Joint monitor (`joint_monitors.py`)

| Class | Condition |
|---|---|
| `JointPositionReached` | `abs(current - target) < threshold` for a single `ActiveConnection1DOF`; uses `shortest_angular_distance` for continuous revolute joints |

## Related

- **Uses:** [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- **Used by:** [[concept.motion-statechart]]

## Provenance

- `monitors.py:1-81` — `ThreadedPayloadMonitor`, `LocalMinimumReached`.
- `payload_monitors.py:1-107` — `CheckControlCycleCount`, `CountControlCycles`, `CountSeconds`, `Pulse`, `Print`.
- `cartesian_monitors.py:1-232` — all geometry monitors.
- `joint_monitors.py:1-41` — `JointPositionReached`.
