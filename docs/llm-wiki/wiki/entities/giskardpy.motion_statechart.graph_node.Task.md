---
id: giskardpy.motion_statechart.graph_node.Task
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/graph_node.py
    lines: [885, 897]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
  - giskardpy.qp.constraint_collection.ConstraintCollection
used_by:
  - pycram.robot_plans.BaseMotion
  - bridge.pycram-giskardpy
  - concept.motion-statechart
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - pycram.motion_executor.MotionExecutor
  - giskardpy.motion_statechart.tasks
status: stable
tags: [giskardpy, task, motion-statechart, qp, constraint, weight]
last_ingest: 2026-05-18
---

_A `MotionStatechartNode` that adds motion constraints (QP goals); returned by every `BaseMotion._motion_chart` implementation in pycram._

## Purpose

`Task` is the leaf node type in the giskardpy Motion State Chart (MSC) graph that carries motion
constraints. Constraints are encoded as `ConstraintCollection` artifacts produced in `build()`
and fed into the QP solver to compute joint velocities. Every `BaseMotion` subclass in pycram
produces exactly one `Task` instance from its `_motion_chart` property.

`Task` itself adds only a `weight: float` field on top of `MotionStatechartNode`. All lifecycle
machinery (start/pause/end/reset conditions, `LifeCycleVariable`, `ObservationVariable`) is
inherited from `MotionStatechartNode`.

## Key attributes

| Name | Kind | Notes |
|---|---|---|
| `weight` | `float` | Priority relative to other tasks. Lower weight loses when constraints compete. Default: `DefaultWeights.WEIGHT_BELOW_CA`. |
| `plot_specs` | `NodePlotSpec` | Visualization style (task-specific, set `init=False`). |

All `MotionStatechartNode` attributes (`name`, `index`, `_start_condition`, `_end_condition`, etc.) are inherited.

## Construction

```python
@dataclass
class MyMotion(BaseMotion):
    target: Pose

    @property
    def _motion_chart(self) -> Task:
        return CartesianPosition(root_link=..., tip_link=..., goal_pose=self.target)
```

Concrete `Task` subclasses used in pycram's gripper motions: `CartesianPosition`,
`CartesianPose`, `JointPositionList`, `Sequence` (a composite task).

## Related

- Parent class: [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- Producer: [[pycram.robot_plans.BaseMotion]] (via `_motion_chart`)
- Bridge: [[bridge.pycram-giskardpy]]
- Executor: [[pycram.motion_executor.MotionExecutor]]

## Open questions

- `MotionStatechart` assembly, the QP controller, and how `Task.build()` constraints feed into
  the solver are not yet documented. Phase 7 will cover giskardpy internals.

## Provenance

- `giskardpy/src/giskardpy/motion_statechart/graph_node.py:885-897` at commit `0528d8cf3` —
  `Task` class definition.
- Lines 362–878 at same commit — `MotionStatechartNode` base class (lifecycle machinery).
