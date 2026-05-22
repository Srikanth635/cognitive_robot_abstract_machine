---
id: pycram.plans.Designator
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/designator.py
    lines: [19, 82]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram
  - concept.designator
  - pycram.plans.PlanNode
  - pycram.plans.Plan
  - pycram.datastructures.Context
used_by:
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.BaseMotion
  - pycram.plans.DesignatorNode
  - pycram.plans.factories.make_node
  - pycram.plans.Plan
status: stable
tags: [designator, base-class, dataclass]
last_ingest: 2026-05-17
---

_Abstract base dataclass for all designators. Provides plan-context delegation and field/parameter introspection._

## Purpose

`Designator` is the shared root of the [[concept.designator]] hierarchy. Concrete
designators (actions and motions) inherit from it to gain:

1. **Context delegation** — `plan`, `robot`, `world`, `context` properties that
   forward to the wrapping [[pycram.plans.DesignatorNode]], raising
   `ContextIsUnavailable` if the designator has not been wrapped yet.
2. **Parameter introspection** — `fields` (classmethod-property) and
   `designator_parameter` (dict of name→value) for the subclass-defined fields,
   excluding the inherited `plan_node`.
3. **Type-hint resolution** — `get_type_hints` resolves the type hints of `__init__`
   in the subclass's module globals.

It is not meant to be subclassed directly outside the `actions` / `motions` trees;
[[pycram.plans.factories.make_node]] only dispatches on those two subtrees.

## When to use

- **Read** this class when introducing a new top-level designator subtree (rare) or
  when debugging why a designator can't access its plan/robot/world.
- **Do not** instantiate directly — it is an abstract base; instantiate a concrete
  `ActionDescription` or `BaseMotion` subclass.

## Construction

```python
# Abstract — instantiated indirectly via concrete subclasses.
# After construction, the designator has no plan_node yet:
my_action.plan_node           # → None
my_action.plan                # → raises ContextIsUnavailable
# The wrapper sets the back-reference:
ActionNode(designator=my_action)  # DesignatorNode.__post_init__ assigns plan_node
my_action.plan                # → the Plan instance
```

## Key attributes

| Name | Type | Notes |
|---|---|---|
| `plan_node` | `Optional[PlanNode]` | Field, `kw_only`, `init=False`, `default=None`, `repr=False`. Set by [[pycram.plans.DesignatorNode]]`.__post_init__`. |
| `plan` *(property)* | `Plan` | Delegates via `plan_node`. Raises `ContextIsUnavailable` if unattached. |
| `robot` *(property)* | `AbstractRobot` (sdt) | Via `self.plan.robot`. |
| `world` *(property)* | `World` (sdt) | Via `self.plan_node.plan.world`. |
| `context` *(property)* | `Context` | Via `self.plan.context`. |
| `fields` *(classmethod, property)* | `List[Field]` | Only this class's own fields (parent `plan_node` removed). Resolves type hints. |
| `designator_parameter` *(property)* | `Dict[str, Any]` | `{f.name: getattr(self, f.name) for f in self.fields}`. |
| `get_type_hints` *(classmethod)* | `Dict[str, Any]` | Resolves `__init__` type hints using the subclass's module globals. |

## Subclasses (direct)

- [[pycram.robot_plans.ActionDescription]] — actions
- [[pycram.robot_plans.BaseMotion]] — motions

## Related

- Concept: [[concept.designator]]
- Wrapper: [[pycram.plans.DesignatorNode]]
- Factory: [[pycram.plans.factories.make_node]]
- Failure type: `pycram.exceptions.ContextIsUnavailable` (entity page not yet created)

## Open questions

- `Designator.fields` uses the `@classmethod` + `@property` stack (lines 55-57 at
  commit `0528d8cf3`). This combination was deprecated in Python 3.11 and **removed
  in 3.13**. If the project's supported Python range crosses 3.13, this will break.
  Worth verifying `pyproject.toml` requirement on the next ingest.

## Provenance

- `pycram/src/pycram/plans/designator.py:19-82` at commit `0528d8cf3` — entire class
  definition.
