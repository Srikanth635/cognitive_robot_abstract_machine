---
id: bridge.pycram-giskardpy
kind: bridge
package: cross
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/base.py
    lines: [22, 63]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/plans/plan_node.py
    lines: [364, 454]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - giskardpy.motion_statechart.graph_node.Task
  - pycram.motion_executor.MotionExecutor
  - giskardpy.qp.qp_controller
  - concept.motion-statechart
  - concept.qp-controller
used_by: []
status: stable
tags: [bridge, pycram, giskardpy, motion-statechart, task, qp, executor]
last_ingest: 2026-05-17
---

_Runtime interface between pycram and giskardpy: motion designators produce Tasks; ActionNode executes the resulting Motion State Chart._

## Purpose

pycram binds to giskardpy through the Motion State Chart (MSC) execution pipeline:

1. **Motion → Task** — every `BaseMotion` subclass implements `_motion_chart` as an abstract
   property that returns a `giskardpy.motion_statechart.graph_node.Task`. This Task encodes one
   QP goal.
2. **ActionNode → MSC execution** — `ActionNode._perform()` calls `collect_motions()` to harvest
   all descendant `MotionNode` Tasks, then passes them to `MotionExecutor.execute_motion_state_chart`.
   The `MotionExecutor` compiles the Tasks into a single giskardpy Motion State Chart and runs it.
3. **Alternative motion mapping** — `BaseMotion.motion_chart` (the public property) can swap the
   motion for an `AlternativeMotion` registered for the current robot type via
   `AlternativeMotion.check_for_alternative`. The alternative inherits the original's `plan_node`.

## Coupling inventory

| Site | pycram side | giskardpy side | Kind |
|---|---|---|---|
| `BaseMotion._motion_chart` | `pycram.robot_plans.BaseMotion` | `giskardpy.motion_statechart.graph_node.Task` | abstract return type |
| `ActionNode.execute_motion_state_chart` | `pycram.plans.ActionNode` | giskardpy MSC runtime (via `MotionExecutor`) | execution call |
| `MotionExecutor` | `pycram.motion_executor.MotionExecutor` | giskardpy MSC construction + execution | delegation |

## Key observations

- The coupling is **one-directional**: giskardpy does not import pycram. pycram's `BaseMotion`
  hierarchy is the only place that creates `Task` objects.
- `MotionNode._perform()` calls `self.motion.perform()` which is a no-op for all built-in motions
  (`perform() = return`). Real execution is entirely through the `ActionNode` → `MotionExecutor`
  path. A motion subclass that overrides `perform()` with real code would bypass the MSC pipeline.

## Related

- Motion base: [[pycram.robot_plans.BaseMotion]]
- Task node: [[giskardpy.motion_statechart.graph_node.Task]]
- MSC executor: [[pycram.motion_executor.MotionExecutor]]
- ActionNode orchestrator: [[pycram.plans.ActionNode]]
- Symmetric bridge: [[bridge.pycram-sdt]]

## QP pipeline (now resolved)

`MotionExecutor` builds a `MotionStatechart`, populates it with the harvested `Task` nodes, calls `msc.compile(context)`, then runs `Executor.tick_until_end()`. The `Executor` owns a `QPController` that:
1. Receives the merged `ConstraintCollection` from all RUNNING nodes.
2. Evaluates it numerically each tick via a compiled CasADi function.
3. Passes to a `QPSolver` (default `QPSolverPIQP`) → gets `xdot`.
4. Converts `xdot` to joint velocity commands and writes them to `WorldState`.

See [[concept.qp-controller]] and [[giskardpy.qp.qp_controller]] for details.

## Provenance

- `pycram/src/pycram/robot_plans/motions/base.py:22-63` at commit `0528d8cf3` — `BaseMotion._motion_chart` abstract property.
- `pycram/src/pycram/plans/plan_node.py:364-454` at commit `0528d8cf3` — `ActionNode._perform()` collect+execute pipeline.
