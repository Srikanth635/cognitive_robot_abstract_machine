---
id: pycram.plans.plan_callbacks.PlanCallback
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_callbacks.py
    lines: [1, 14]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.PlanEntity
  - pycram.plans.PlanNode
used_by:
  - pycram.plans.Plan
status: stable
tags: [callback, hook, plan-entity, lifecycle]
last_ingest: 2026-05-17
---

_Lifecycle hook attached to a [[pycram.plans.Plan]]; receives `on_start` and `on_end` notifications for every [[pycram.plans.PlanNode]] that executes._

## Purpose

`PlanCallback` is the standard extension point for cross-cutting concerns (logging, telemetry, UI updates) that need to observe every node transition without being part of the plan logic. It extends [[pycram.plans.PlanEntity]] so that `Plan.add_plan_entity(callback)` registers it alongside the `Context`.

## When to use

- **Logging execution timing** — override `on_start`/`on_end` to timestamp every node.
- **Telemetry / tracing** — record which nodes were executed and in what order.
- **Not for:** altering plan logic or raising failures — callbacks are notification-only.

## Construction

```python
class MyCallback(PlanCallback):
    def on_start(self, node: PlanNode):
        print(f"Starting {node}")

    def on_end(self, node: PlanNode):
        print(f"Finished {node}")

plan.add_plan_entity(MyCallback())
```

## Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `on_start` | `(node: PlanNode) → None` | Called when a node begins `perform()` |
| `on_end` | `(node: PlanNode) → None` | Called when a node exits `perform()` (success or failure) |

Default implementations are no-ops (`...`).

## Related

**Uses:** [[pycram.plans.PlanEntity]], [[pycram.plans.PlanNode]]

**Used by:** [[pycram.plans.Plan]]

## Provenance

- `pycram/src/pycram/plans/plan_callbacks.py` lines 1–14 (commit `0528d8cf3`) — full `PlanCallback` dataclass.
