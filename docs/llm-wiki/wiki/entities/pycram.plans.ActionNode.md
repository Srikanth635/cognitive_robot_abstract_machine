---
id: pycram.plans.ActionNode
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_node.py
    lines: [364, 454]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.DesignatorNode
  - pycram.robot_plans.ActionDescription
  - pycram.plans.MotionNode
  - pycram.plans.failures.PlanFailure
  - pycram.motion_executor.MotionExecutor
  - pycram.datastructures.ExecutionData
used_by:
  - pycram.plans.UnderspecifiedNode
  - pycram.plans.MotionNode
  - pycram.plans.factories.make_node
status: stable
tags: [plan-node, action, execution, motion-state-chart]
last_ingest: 2026-05-17
---

_Concrete plan-graph node wrapping an [[pycram.robot_plans.ActionDescription]]. Drives the full execution cycle: pre-state capture → action body → Motion State Chart construction + execution → post-state capture._

## Purpose

`ActionNode` is the **execution unit for actions** in the plan graph. It is a
`DesignatorNode` holding an `ActionDescription` as its designator. When `_perform()`
fires, it orchestrates a four-step sequence:

1. **Pre-capture** (`create_execution_data_pre_perform`): snapshot robot pose and world
   state, and record the current world-modification-block watermark.
2. **Action body** (`action.perform()`): runs the `ActionDescription`'s `perform()`
   method, which builds the action's subplan (child `MotionNode`s and nested
   `ActionNode`s) and evaluates preconditions.
3. **MSC construction + execution** (`execute_motion_state_chart`): collects all
   direct-descendant `MotionNode`s, bundles their `BaseMotion.motion_chart` giskard
   `Task`s into a `MotionExecutor`, and calls `MotionExecutor.execute()`.
4. **Post-capture** (`update_execution_data_post_perform`): records the robot's
   end pose and new world-modification blocks for the `ExecutionData` record.

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `execution_data` | `ExecutionData` | Populated by pre/post capture. Holds pose snapshots and world-mod blocks. |
| `motion_executor` | `MotionExecutor` | Built and executed by `execute_motion_state_chart()`. `None` until first perform. |
| `_world_modification_block_length_pre_perform` | `Optional[int]` | *Declared* field. Runtime writes `_last_world_modification_block_pre_perform_index` instead — naming inconsistency (see Open questions). |
| `action` *(property)* | `ActionDescription` | Typed alias to `self.designator`. |

## Execution flow

```
_perform()
  ↳ create_execution_data_pre_perform()   # snapshot pre-state
  ↳ action.perform()                      # ActionDescription builds subplan
  ↳ execute_motion_state_chart()
      ↳ collect_motions()                 # descendant MotionNodes where parent_action_node == self
      ↳ MotionExecutor(motions, world, ros_node, plan_node)
      ↳ motion_executor.construct_msc()
      ↳ motion_executor.execute()
  ↳ update_execution_data_post_perform()  # snapshot post-state
```

## Motion collection rule

`collect_motions()` returns `[node.motion.motion_chart for node in self.descendants if isinstance(node, MotionNode) and self is node.parent_action_node]`.

The `parent_action_node` guard is critical: it ensures that motions belonging to
**nested actions** are not consumed by an outer action. Each action runs only its own
direct-descendant motions.

## Related

- Abstract base: [[pycram.plans.DesignatorNode]]
- Wraps: [[pycram.robot_plans.ActionDescription]]
- Collects motions from: [[pycram.plans.MotionNode]]
- Executes via: [[pycram.motion_executor.MotionExecutor]] (stub)
- Records into: [[pycram.datastructures.ExecutionData]] (stub)
- Created by: [[pycram.plans.UnderspecifiedNode]] (grounding loop) and [[pycram.plans.factories.make_node]] (fully specified path)

## Open questions

- The declared field is `_world_modification_block_length_pre_perform` but the code
  writes `_last_world_modification_block_pre_perform_index`. Either a rename-in-progress
  or a latent bug. Flagged in [[pycram.plans.DesignatorNode]]; check during Phase 4 ingest.
- `MotionExecutor` receives `ros_node=self.plan.context.ros_node`, so `context` must
  be non-`None` at perform time — consistent with the guard risk noted on [[pycram.plans.Plan]].

## Provenance

- `pycram/src/pycram/plans/plan_node.py:364-454` at commit `0528d8cf3`.
