---
id: pycram.plans.MotionNode
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_node.py
    lines: [457, 488]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.DesignatorNode
  - pycram.robot_plans.BaseMotion
  - pycram.plans.ActionNode
used_by:
  - pycram.plans.ActionNode
  - pycram.plans.factories.make_node
status: stable
tags: [plan-node, motion, motion-state-chart, leaf]
last_ingest: 2026-05-17
---

_Concrete plan-graph leaf node wrapping a [[pycram.robot_plans.BaseMotion]]. Not executed directly — aggregated by its nearest ancestor [[pycram.plans.ActionNode]] into a Motion State Chart._

## Purpose

`MotionNode` is the **leaf node for motions**. Unlike `ActionNode`, it does not drive
execution: its `_perform()` delegates to `self.motion.perform()`, but in normal plan
execution the motion is consumed *before* `_perform()` is called because
`ActionNode.collect_motions()` harvests all direct-descendant `MotionNode`s when
constructing the MSC.

The `parent_action_node` property is the key link: it walks `self.path` upward and
returns the nearest `ActionNode` ancestor. `ActionNode.collect_motions()` filters by
this property to claim only its own motions, leaving nested-action motions to their
respective parent.

## When to use / read

- Read when understanding how `BaseMotion` goals become `giskardpy` `Task` objects
  in the MSC.
- Don't construct `MotionNode` directly — [[pycram.plans.factories.make_node]] creates
  it from a `BaseMotion` instance.

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `motion` *(property)* | `BaseMotion` | Typed alias to `self.designator`. |
| `parent_action_node` *(property)* | `Optional[ActionNode]` | First `ActionNode` found walking `self.path` upward. `None` if not inside an action subtree. |

## Execution relationship

```
ActionNode._perform()
  └── collect_motions()
        └── [node.motion.motion_chart
             for node in self.descendants
             if isinstance(node, MotionNode)
             and self is node.parent_action_node]
              → List[giskardpy.motion_statechart.graph_node.Task]
                  → passed to MotionExecutor
```

`MotionNode._perform()` runs only in the fallback path (e.g. isolated replay). In
normal plan execution the `ActionNode` drives the MSC.

## Related

- Abstract base: [[pycram.plans.DesignatorNode]]
- Wraps: [[pycram.robot_plans.BaseMotion]]
- Consumed by: [[pycram.plans.ActionNode]] via `collect_motions()`
- Created by: [[pycram.plans.factories.make_node]]

## Provenance

- `pycram/src/pycram/plans/plan_node.py:457-488` at commit `0528d8cf3`.
