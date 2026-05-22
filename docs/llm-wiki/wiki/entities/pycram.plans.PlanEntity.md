---
id: pycram.plans.PlanEntity
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_entity.py
    lines: [10, 16]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram
  - pycram.plans.Plan
used_by:
  - pycram.plans.PlanNode
  - pycram.plans.Plan
  - pycram.datastructures.Context
  - pycram.plans.plan_callbacks.PlanCallback
status: stable
tags: [plan, base-class, dataclass]
last_ingest: 2026-05-18
---

_Base dataclass for anything that lives inside a `Plan` — including [[pycram.plans.PlanNode]] and the runtime `Context` — providing a single back-reference to its owning plan._

## Purpose

`PlanEntity` is the minimal surface that everything inside a plan shares: a single
optional `plan` field. Two distinct entity families inherit from it:

- **[[pycram.plans.PlanNode]]** (and its subclasses) — graph nodes.
- **`Context`** — the runtime configuration object attached to a plan
  (`Plan.add_plan_entity(self.context)` in `Plan.__post_init__`). Context page is a
  stub at [[pycram.datastructures.Context]].

Membership in a plan is managed via `Plan.add_plan_entity(entity)` / `remove_plan_entity`,
which simply assigns or nulls the back-reference.

## When to use / read

- Read when you want to understand why both `PlanNode` and `Context` participate in
  the plan's lifecycle.
- Subclass only if you are adding a *new kind of thing* that should be owned by a
  plan but is not itself a graph node. This is rare.

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `plan` | `Optional[Plan]` | Field, `kw_only`, `default=None`. Set by `Plan.add_plan_entity`. |

That's the whole class (7 lines of body at the cited source).

## Related

- Subclasses: [[pycram.plans.PlanNode]], [[pycram.datastructures.Context]] (stub).
- Container: [[pycram.plans.Plan]].
- Concept: [[concept.designator]].

## Provenance

- `pycram/src/pycram/plans/plan_entity.py:10-16` at commit `0528d8cf3` — full class.
