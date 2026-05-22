---
id: pycram.plans.UnderspecifiedNode
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_node.py
    lines: [301, 343]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.PlanNode
  - pycram.robot_plans.ActionDescription
  - pycram.plans.ActionNode
  - krrood.entity_query_language.backends.QueryBackend
used_by:
  - concept.krrood-eql
  - pycram.plans.factories.make_node
status: stable
tags: [plan-node, underspecified, krrood, grounding]
last_ingest: 2026-05-17
---

_A [[pycram.plans.PlanNode]] that wraps a krrood `underspecified(...)` expression — an action type not yet bound to concrete values. Performs by iterating query-backend solutions, attaching each as a child [[pycram.plans.ActionNode]], until one succeeds or the iterator is exhausted._

## Purpose

`UnderspecifiedNode` bridges **symbolic action specifications** (krrood `Match` objects)
and **concrete execution**. It is created by [[pycram.plans.factories.make_node]] when
`is_underspecified(action_like)` is true.

On `_perform()`:
1. **Lazy initialization**: on first call, evaluates
   `plan.context.query_backend.evaluate(underspecified_action)` to get an
   `Iterator[ActionDescription]`.
2. **Grounding loop**: for each solution `grounded_action`, creates
   `ActionNode(designator=grounded_action)`, attaches it as a child via `add_child`,
   and calls `new_child.perform()`.
3. **Try-until-success**: if `perform()` raises `PlanFailure`, continues with the next
   solution. If `perform()` succeeds, returns.
4. If the iterator is exhausted without success, the function returns without a result
   (no explicit `PlanFailure` raised — see Open questions).

A `limit` clause on the `Match` controls iterator length, bounding the number of
grounding attempts.

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `underspecified_action` | `krrood.entity_query_language.query.match.Match` | The krrood `Match` expression. `kw_only`. |
| `_action_iterator` | `Optional[Iterator[ActionDescription]]` | Lazily created on first `_perform()`. Persists across calls, enabling retry. |
| `designator_type` *(property)* | `Type` | `underspecified_action.type` — the target designator class. Used by `__repr__`. |

## Grounding flow

```
factories.make_node(action_like)         # is_underspecified → True
  → UnderspecifiedNode(underspecified_action=action_like)
      ↓ _perform()
      ↓ query_backend.evaluate(match) → Iterator[ActionDescription]
      ↓ for grounded_action:
          ActionNode(designator=grounded_action) → add_child → new_child.perform()
          except PlanFailure: continue
          return (success)
```

## Related

- Base class: [[pycram.plans.PlanNode]]
- Creates children: [[pycram.plans.ActionNode]]
- Grounded type: [[pycram.robot_plans.ActionDescription]]
- Created by: [[pycram.plans.factories.make_node]]
- Runtime dependency: `plan.context.query_backend` (part of [[pycram.datastructures.Context]] stub)

## Open questions

- On iterator exhaustion, `_perform()` returns silently (no `PlanFailure` raised).
  Whether callers treat a no-result return as failure is unclear from this file.
  Worth confirming during Phase 3 ingest of `pycram.language`.
- The `limit` clause lives in the krrood `Match` object and is enforced by the query
  backend — not visible in this class. Verify when krrood entities are ingested.

## Provenance

- `pycram/src/pycram/plans/plan_node.py:301-343` at commit `0528d8cf3`.
