---
id: pycram.plans.PlanNode
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_node.py
    lines: [39, 299]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.PlanEntity
  - pycram.plans.Plan
  - pycram.plans.failures.PlanFailure
  - concept.designator
used_by:
  - pycram.plans.Plan
  - pycram.plans.DesignatorNode
  - pycram.plans.UnderspecifiedNode
  - pycram.plans.factories.make_node
  - pycram.language.LanguageNode
  - pycram.robot_plans.ActionDescription
  - pycram.motion_executor.MotionExecutor
  - pycram.plans.Designator
  - pycram.plans.plan_callbacks.PlanCallback
status: stable
tags: [plan, graph-node, abstract-base, task-status]
last_ingest: 2026-05-18
---

_Abstract graph-node base class. Lives in a `rustworkx` directed graph; carries execution status, timing, result, and parent/child navigation._

## Purpose

`PlanNode` is the abstract base for every node in a `Plan`'s `plan_graph`. It extends
[[pycram.plans.PlanEntity]] with everything needed to live in a graph and carry
execution state:

- **Identity in the graph**: `index` (its node id in the rustworkx graph),
  `layer_index` (its ordered position among siblings — rustworkx itself doesn't
  preserve child order, so the field carries it).
- **Execution state**: `status: TaskStatus`, `start_time`, `end_time`, `reason`
  (a `PlanFailure` if it failed), `result`.
- **Navigation properties**: `parent`, `children`, `descendants`, `path`, `depth`,
  `siblings`, `left_siblings`, `right_siblings`, `left_neighbour`, `right_neighbour`,
  `previous_nodes`, `is_leaf`.
- **Lifecycle methods**: `perform()` (the public entry; manages status and exceptions),
  `_perform()` (abstract subclass hook), `interrupt()`, `pause()`, `resume()`,
  `mount_subplan(root)`, `add_child(child)`, `simplify()` (no-op by default).
- **Designator queries**: `get_previous_node_by_designator_type(*type_)`.

This page documents the abstract base. Concrete subclasses each get their own page
(see "Subclasses" below).

## When to use / read

- Read when you need to understand how nodes navigate the plan graph or how
  execution state propagates (e.g. interruption checks via `is_interrupted`).
- Subclass only when adding a new **kind** of node (not when adding new actions or
  motions — those subclass `Designator`, not `PlanNode`). The major existing
  subclasses are listed below.

## Lifecycle (the only public entry: `perform()`)

```
status = RUNNING
try:
    result = self._perform()              # subclass-implemented
except PlanFailure as e:
    status = FAILED; reason = e; raise
finally:
    end_time = datetime.now()
status = SUCCEEDED
```

Before any of that, `perform()` walks `self.path` and, if any parent is
`INTERRUPTED`, sets this node's status to `INTERRUPTED` and returns immediately —
this is how interruption propagates downward.

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `status` | `TaskStatus` | Default `CREATED`. Mutated by `perform`, `interrupt`, `pause`, `resume`. |
| `start_time` | `Optional[datetime]` | Defaults to `datetime.now()`. |
| `end_time` | `Optional[datetime]` | Set on `perform()` completion. |
| `reason` | `Optional[PlanFailure]` | Set when `_perform` raises. |
| `result` | `Optional[Any]` | Return value of `_perform`. |
| `index` | `Optional[int]` | rustworkx node id; assigned by `Plan.add_node`. |
| `layer_index` | `Optional[int]` | Position among siblings. Set by `Plan.add_edge`. |

## Subclasses (concrete)

- [[pycram.plans.DesignatorNode]] (abstract) — wraps a Designator.
  - [[pycram.plans.ActionNode]] — wraps `ActionDescription`; drives execution + MSC.
  - [[pycram.plans.MotionNode]] — wraps `BaseMotion`; passive leaf harvested by parent `ActionNode`.
- [[pycram.plans.UnderspecifiedNode]] — wraps a krrood `Match` expression; grounds
  it lazily via the query backend into `ActionNode` children until one succeeds.
- The `pycram.language.*Node` family (`SequentialNode`, `ParallelNode`,
  `RepeatNode`, `MonitorNode`, `TryInOrderNode`, `TryAllNode`, `CodeNode`,
  `LanguageNode` abstract base) — combinator nodes, covered in Phase 3.

## Related

- Parent: [[pycram.plans.PlanEntity]].
- Container: [[pycram.plans.Plan]].
- Subclass family: [[pycram.plans.DesignatorNode]].
- Constructed by: [[pycram.plans.factories.make_node]].
- Failure type: `pycram.plans.failures.PlanFailure` (stub: [[pycram.plans.failures.PlanFailure]]).
- Status enum: `pycram.datastructures.enums.TaskStatus` (not yet a page).

## Open questions

- `simplify()` is overridden on [[pycram.language.LanguageNode]] to flatten same-type
  children (confirmed Phase 3 ingest). The base no-op is correct for `ActionNode`,
  `MotionNode`, and `UnderspecifiedNode`. Resolved.
- `pause()` sets `status = TaskStatus.PAUSE` (singular) while `is_paused` checks
  for the same enum value — consistent locally but worth confirming the rest of
  the codebase agrees (some enums use `PAUSED`).

## Provenance

- `pycram/src/pycram/plans/plan_node.py:39-299` at commit `0528d8cf3` — `PlanNode` base.
- Lines 301-308: `UnderspecifiedNode` class header (full body to be expanded later).
