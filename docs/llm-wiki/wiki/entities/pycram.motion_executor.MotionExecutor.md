---
id: pycram.motion_executor.MotionExecutor
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/motion_executor.py
    lines: [32, 153]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - giskardpy.motion_statechart.graph_node.Task
  - giskardpy.motion_statechart.graph_node.EndMotion
  - giskardpy.motion_statechart.context
  - sdt.world.World
  - pycram.plans.PlanNode
used_by:
  - bridge.pycram-giskardpy
  - pycram.plans.ActionNode
  - pycram.alternative_motion_mapping.AlternativeMotion
status: stable
tags: [motion, executor, motion-statechart, giskardpy, simulation, real-robot]
last_ingest: 2026-05-17
---

_Bridge between pycram's motion designators and giskardpy's tick loop: wraps a list of [[giskardpy.motion_statechart.graph_node.Task|Task]] objects in a [[giskardpy.motion_statechart.motion_statechart.MotionStatechart|MotionStatechart]], compiles, and runs it to completion (or until the parent [[pycram.plans.PlanNode]] is interrupted/paused)._

## Purpose

`MotionExecutor` is the final hand-off from pycram to giskardpy. An [[pycram.plans.ActionNode]] collects one `Task` per child `MotionNode`, hands them to a `MotionExecutor`, and calls `execute()`. The executor wraps all tasks in a giskardpy `Sequence` node, adds an `EndMotion` sentinel, and then drives the tick loop either through `Ros2Executor` (simulation or real robot) or skips execution entirely when `ExecutionType.NO_EXECUTION` is set.

The `execution_type` class variable is a global mode switch set by `ExecutionEnvironment` context managers (`simulated_robot`, `real_robot`, `no_execution`).

## When to use

- **Invoked by** `ActionNode.perform()` — not called directly from user code.
- **To switch execution mode** wrap plan execution in an `ExecutionEnvironment`:
  ```python
  with simulated_robot:
      plan.perform()
  ```

## Construction / dependencies

```python
executor = MotionExecutor(
    motions=[task1, task2],   # List[Task] from MotionNodes
    world=context.world,      # sdt.world.World
    plan_node=action_node,    # PlanNode for interrupt/pause polling
    ros_node=ctx.ros_node,    # Optional ROS 2 node
)
executor.construct_msc()  # builds Sequence + EndMotion inside MotionStatechart
executor.execute()        # dispatches on execution_type
```

## construct_msc / execute flow

`construct_msc()`:
1. Creates a fresh `MotionStatechart()`.
2. Wraps all motions in a giskardpy `Sequence(nodes=self.motions)`.
3. Adds `EndMotion.when_true(sequence_node)` — terminates the MSC when the sequence completes.

`_execute_for_simulation()`:
1. Creates `Ros2Executor` with `MotionStatechartContext(world, QPControllerConfig)`.
2. `executor.compile(self.motion_state_chart)`.
3. Tick loop up to 2000 iterations; checks `plan_node.is_interrupted` and `plan_node.is_paused` each cycle.
4. On `is_end_motion()` — breaks.
5. Cleanup: zero velocities, `cleanup_nodes()`, `context.cleanup()`.
6. If not done → raises `MotionDidNotFinish(failed_nodes)`.

`_execute_for_real()`:
- Uses `GiskardWrapper` (ROS 2 python interface); interrupt monitoring is delegated to a side thread.

## ExecutionEnvironment

A companion `@dataclass` that saves/restores `MotionExecutor.execution_type` on enter/exit, enabling nested environments:

```python
simulated_robot = ExecutionEnvironment(ExecutionType.SIMULATED)
real_robot      = ExecutionEnvironment(ExecutionType.REAL)
no_execution    = ExecutionEnvironment(ExecutionType.NO_EXECUTION)
```

## Related

**Uses:** [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]], [[giskardpy.motion_statechart.graph_node.Task]], [[sdt.world.World]]

**Used by:** [[pycram.plans.ActionNode]]

**See also:** [[bridge.pycram-giskardpy]]

## Open questions

- The tick loop runs up to 2000 iterations at target 50 Hz → max 40 s. Is this hard-coded or configurable? `QPControllerConfig.target_frequency=50` is set inline — no per-plan override seen.

## Provenance

- `pycram/src/pycram/motion_executor.py` lines 32–153 (commit `0528d8cf3`) — `MotionExecutor` dataclass, `construct_msc`, `execute`, `_execute_for_simulation`, `_execute_for_real`.
- `pycram/src/pycram/motion_executor.py` lines 156–202 — `ExecutionEnvironment` + module-level `simulated_robot`/`real_robot`/`no_execution` singletons.
