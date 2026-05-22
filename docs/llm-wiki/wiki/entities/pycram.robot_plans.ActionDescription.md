---
id: pycram.robot_plans.ActionDescription
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/base.py
    lines: [41, 137]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.Designator
  - pycram.plans.PlanNode
  - pycram.plans.Plan
  - pycram.datastructures.Context
  - pycram.plans.failures.PlanFailure
  - concept.designator
  - krrood.entity_query_language.core.variable.Variable
  - krrood.entity_query_language.predicate.Predicate
used_by:
  - concept.krrood-eql
  - pycram.plans.factories.make_node
  - pycram.plans.ActionNode
  - pycram.plans.UnderspecifiedNode
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.core.NavigateAction
  - pycram.robot_plans.actions.composite
  - pycram.robot_plans.actions.core.container
  - pycram.robot_plans.actions.core.misc
  - pycram.robot_plans.actions.core.robot_body
status: stable
tags: [designator, action, abstract-base, krrood, symbolic]
last_ingest: 2026-05-18
---

_Abstract subclass of [[pycram.plans.Designator]] for actions: a "builder for plans" with symbolic pre/post conditions._

## Purpose

`ActionDescription` is the abstract base for **all actions** in pycram. An action
describes high-level robot behavior parametrically; when its plan node is performed,
the action's `execute()` method builds a (sub)plan of motions and/or other actions.
Actions also carry **symbolic pre/post conditions** expressed via
`krrood.entity_query_language`, evaluated against the current `Context`.

In the [[concept.designator]] vocabulary: actions are non-leaf — they can spawn
subplans — while [[pycram.robot_plans.BaseMotion|motions]] are leaf-level.

## When to use

- **Subclass** when adding a new high-level robot behavior (e.g. `PickUp`, `Transport`,
  `ReachAction`). Subclasses define their parameters as dataclass fields and implement
  `execute()` to assemble subplans (typically via [[pycram.plans.factories.make_node|the
  plan factories]]).
- **Override** `pre_condition` / `post_condition` to declare symbolic guards. Both
  receive `(variables, context, kwargs)` and return a `SymbolicExpression` (default:
  `True`).
- **Do not** put low-level motion code here — actions assemble *plans*; motions assemble
  *goals*.

## Construction

```python
@dataclass
class MyAction(ActionDescription):
    target: Pose
    arm: Arms
    def execute(self) -> Any:
        # Build a subplan from other actions or motions.
        ...
    @staticmethod
    def pre_condition(variables, context, kwargs):
        # Optional symbolic guard.
        return and_(...)
```

Concrete examples in `pycram/src/pycram/robot_plans/actions/core/`:
`ReachAction`, `PickUp`, `Placing`, … and composites in `.../composite/`
(`Searching`, `Transporting`, …). These entity pages will be created as ingests
cover them.

## Key methods and attributes

| Name | Kind | Notes |
|---|---|---|
| `world` | property → `Optional[World]` | Overrides `Designator.world`; raises `ContextIsUnavailable` if no plan. |
| `perform()` | method | Logs, optionally evaluates pre-condition (gated by `context.evaluate_conditions`), then calls `execute()`. Returns the action's result. |
| `execute()` | abstract method | Subclass hook. Should only build motions and/or other actions. |
| `pre_condition` | static, default `True` | Signature: `(variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]) -> SymbolicExpression`. |
| `post_condition` | static, default `True` | Same signature as `pre_condition`. |
| `bound_variables` | cached_property → `Dict[str, Variable]` | krrood variables for each action field, built via `_create_variables`. |
| `evaluate_pre_condition()` | method | Builds and evaluates `pre_condition`; raises `ConditionNotSatisfied` on failure. |
| `evaluate_post_condition()` | method | Symmetric to the above. |
| `add_subplan(subplan_root)` | method | Migrates a subplan into this action's plan and links it as a child of this action's plan node. |

Two module-level type aliases also defined here:

- `ActionType = TypeVar("ActionType", bound=ActionDescription)`
- `type DescriptionType[T] = Union[Iterable[T], T, ...]` (PEP 695 syntax)

## Related

- Parent: [[pycram.plans.Designator]]
- Sibling: [[pycram.robot_plans.BaseMotion]]
- Wrapper: `ActionNode` (subclass of [[pycram.plans.DesignatorNode]])
- Concept: [[concept.designator]]
- Dispatcher: [[pycram.plans.factories.make_node]]
- External: `krrood.entity_query_language` (symbolic expressions for conditions)

## Open questions

- The class also lives logically under `pycram.robot_plans` thanks to the import
  cascade in `pycram/src/pycram/robot_plans/actions/__init__.py` (1 line, importing
  from `base`?). The `__init__` content was not fully expanded in this ingest —
  confirm public API path on the next ingest.

## Provenance

- `pycram/src/pycram/robot_plans/actions/base.py:41-137` at commit `0528d8cf3` — full
  class definition.
- `pycram/src/pycram/robot_plans/actions/base.py:140-141` — `ActionType` and
  `DescriptionType[T]` aliases.
