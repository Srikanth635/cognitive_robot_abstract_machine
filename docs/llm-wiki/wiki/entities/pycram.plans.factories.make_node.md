---
id: pycram.plans.factories.make_node
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/factories.py
    lines: [126, 147]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.Designator
  - pycram.plans.DesignatorNode
  - pycram.plans.PlanNode
  - pycram.plans.Plan
  - pycram.plans.UnderspecifiedNode
  - pycram.plans.ActionNode
  - pycram.plans.MotionNode
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.BaseMotion
  - concept.designator
used_by:
  - pycram
  - pycram.plans.factories
status: stable
tags: [factory, dispatch, plan, designator]
last_ingest: 2026-05-17
---

_Single-dispatch factory that maps an `action_like` value (PlanNode | ActionDescription | BaseMotion | underspecified krrood expression) to the correct concrete plan-graph node._

## Purpose

`make_node` is the **canonical entry point** for turning a designator (or already-built
node) into a `PlanNode` ready to be added to a `Plan`. It is what the public plan
builders (`sequential`, `parallel`, `try_in_order`, `try_all`, `monitor`, `repeat`,
`code`, `execute_single`) call to wrap their children.

It is also the place that **enforces the two-subtree contract** of the [[concept.designator]]:
only `ActionDescription`, `BaseMotion`, or already-built `PlanNode` / underspecified
expressions are accepted; anything else hits `assert_never`.

## When to use

- **Read** when adding a new public plan combinator or a new top-level Designator
  subtree — both require thinking about the dispatch here.
- **Don't call** directly in user-facing plan code; use the higher-level factories
  (`sequential`, etc.) from the same module.

## Signature and dispatch

```python
def make_node(action_like: ActionLike) -> PlanNode:
    if isinstance(action_like, PlanNode):
        return action_like
    elif is_underspecified(action_like):
        return UnderspecifiedNode(underspecified_action=action_like)
    elif isinstance(action_like, ActionDescription):
        return ActionNode(designator=action_like)
    elif isinstance(action_like, BaseMotion):
        return MotionNode(designator=action_like)
    else:
        assert_never(action_like)
```

The `ActionLike` type alias is imported from `pycram.plans.plan_node` (not expanded
in this ingest).

## Related

- Companions in the same module: `execute_single`, `sequential`, `parallel`,
  `try_in_order`, `try_all`, `monitor`, `repeat`, `code` — all call `make_node` on
  each child. Not yet promoted to their own pages.
- Dispatch targets:
  - `ActionNode` ← [[pycram.robot_plans.ActionDescription]]
  - `MotionNode` ← [[pycram.robot_plans.BaseMotion]]
  - `UnderspecifiedNode` — uses `krrood.entity_query_language.query.match.is_underspecified`.
- Concept: [[concept.designator]]
- Wrapper base: [[pycram.plans.DesignatorNode]]

## Open questions

- `ActionLike = Union[Match, Designator, PlanNode]` (defined at end of `plan_node.py:489`).
  `assert_never` catches anything outside this union. Resolved in Phase 2/3 ingests.

## Provenance

- `pycram/src/pycram/plans/factories.py:126-147` at commit `0528d8cf3` — full function.
