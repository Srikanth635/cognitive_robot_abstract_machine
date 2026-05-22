---
id: pycram.plans.factories
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/factories.py
    lines: [1, 147]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.language.LanguageNode
  - pycram.plans.Plan
  - pycram.plans.factories.make_node
  - pycram.datastructures.Context
  - concept.plan-language
used_by:
  - pycram
  - concept.plan-language
  - pycram.robot_plans.actions.core.NavigateAction
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
status: stable
tags: [factory, combinator, plan, language]
last_ingest: 2026-05-18
---

_Module of public combinator functions that construct plan trees from `ActionLike` children. The user-facing API for the [[concept.plan-language]]._

## Purpose

`pycram.plans.factories` is the **entry point** for building executable plans. Each
combinator function accepts a list of `ActionLike` children, wraps them in a
`LanguageNode`, attaches the node to a new `Plan`, and returns the root node.

The module also contains [[pycram.plans.factories.make_node]] (the low-level dispatch
from Designator/PlanNode to a concrete node type) and the internal helper
`_make_plan_from_type_and_children`.

## Public API

| Function | Returns | Semantics |
|---|---|---|
| `execute_single(action_like, context)` | `PlanNode` | Wraps one action in a plan; no language node. |
| `sequential(children, context)` | `SequentialNode` | Execute children sequentially; first failure propagates. |
| `parallel(children, context)` | `ParallelNode` | Execute all in parallel; first failure after join. |
| `try_in_order(children, context)` | `TryInOrderNode` | Sequential; continue past failures; fail if all fail. |
| `try_all(children, context)` | `TryAllNode` | Parallel; fail only if all fail. |
| `monitor(children, condition, behavior, context)` | `MonitorNode` | Sequential with live condition monitoring. |
| `repeat(children, repetitions, context)` | `RepeatNode` | Sequential, looped N times. |
| `code(function, context)` | `CodeNode` | Single `Callable`-wrapping node. |

All functions accept an optional `context: Optional[Context]` which is passed to
the `Plan` constructor.

## Internal mechanics (`_make_plan_from_type_and_children`)

```python
plan = Plan(context=context)
plan.add_node(root)                        # root is the LanguageNode instance
for action_like in children:
    child = make_node(action_like)
    if isinstance(child, LanguageNode):
        root.mount_subplan(child)          # migrates the child's plan graph
    else:
        root.add_child(child)
plan.simplify()                            # bottom-up same-type flattening
return root
```

The `mount_subplan` vs `add_child` branch is critical: if a child is itself a
language node (returned from a combinator call), its whole sub-plan graph is
**migrated** into the parent plan. Plain leaf nodes (`ActionNode`, `MotionNode`) are
just added as children.

All `LanguageNode` class imports are **deferred** inside function bodies
(`from pycram.language import SequentialNode`) to avoid a circular import with
`pycram.language`, which itself imports from `pycram.plans.failures`.

## Related

- Language nodes: [[pycram.language.LanguageNode]]
- Dispatch: [[pycram.plans.factories.make_node]]
- Plan: [[pycram.plans.Plan]]
- Concept: [[concept.plan-language]]

## Open questions

- `code(function, context)` routes through `execute_single(root, context)` rather than
  `_make_plan_from_type_and_children`. This means `plan.simplify()` is NOT called for
  `CodeNode`. Presumably intentional (it has no children), but worth noting.

## Provenance

- `pycram/src/pycram/plans/factories.py:1-147` at commit `0528d8cf3`.
