---
id: pycram.plans.DesignatorNode
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/plan_node.py
    lines: [346, 362]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.Designator
  - pycram.plans.PlanNode
  - concept.designator
used_by:
  - pycram.plans.factories.make_node
  - pycram.plans.Plan
  - pycram.plans.ActionNode
  - pycram.plans.MotionNode
status: stable
tags: [plan-node, wrapper, abstract-base]
last_ingest: 2026-05-18
---

_Abstract plan-graph node that wraps a [[concept.designator|Designator]] and wires the back-reference `designator.plan_node = self`. Parent of [[pycram.plans.ActionNode]] and [[pycram.plans.MotionNode]]._

## Purpose

`DesignatorNode` is the bridge between a parametric **Designator** and the **plan
graph**. It extends [[pycram.plans.PlanNode]] with a single `designator` field and a
`__post_init__` that sets `designator.plan_node = self`, completing the bidirectional
link described in [[concept.designator]].

The class itself is abstract (`ABC`). All real work happens in the two concrete
subclasses — each has its own page:

- [[pycram.plans.ActionNode]] — wraps [[pycram.robot_plans.ActionDescription]].
  Performs the action, builds + executes a Motion State Chart.
- [[pycram.plans.MotionNode]] — wraps [[pycram.robot_plans.BaseMotion]].
  Passive leaf; its motion goal is harvested by the parent `ActionNode`.

## When to use / read

- Read when you need to understand the **back-reference wiring** or the abstract
  `__repr__` that both children inherit.
- For execution semantics, go directly to [[pycram.plans.ActionNode]] or
  [[pycram.plans.MotionNode]].
- Don't construct `DesignatorNode` directly — use [[pycram.plans.factories.make_node]].

## Construction

```python
# Both concrete subclasses call __post_init__ implicitly:
node = ActionNode(designator=my_action_description)
my_action_description.plan_node is node   # True — set by __post_init__
```

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `designator` | `Designator` | Field, `kw_only`. The wrapped designator instance. |
| (inherited) | — | All `PlanNode` attributes: `index`, `layer_index`, `status`, `start_time`, `end_time`, `reason`, `result`. |

## Subclasses

| Class | Wraps | Own page |
|---|---|---|
| `ActionNode` | `ActionDescription` | [[pycram.plans.ActionNode]] |
| `MotionNode` | `BaseMotion` | [[pycram.plans.MotionNode]] |

## Related

- Wrapped concept: [[concept.designator]] → [[pycram.plans.Designator]]
- Concrete subclasses: [[pycram.plans.ActionNode]], [[pycram.plans.MotionNode]]
- Factory: [[pycram.plans.factories.make_node]]

## Open questions

- `ActionNode` declares `_world_modification_block_length_pre_perform` but writes
  `_last_world_modification_block_pre_perform_index` at runtime — a naming mismatch.
  Likely a rename-in-progress; check during Phase 4 ingest.

## Provenance

- `pycram/src/pycram/plans/plan_node.py:346-362` at commit `0528d8cf3` — abstract base.
- `ActionNode` body: lines 364-454. `MotionNode` body: lines 457-488.
