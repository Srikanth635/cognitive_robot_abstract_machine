---
id: krrood.entity_query_language.predicate.Predicate
kind: entity
package: cross
source_paths:
  - path: krrood/src/krrood/entity_query_language/predicate.py
    lines: [1, 120]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - krrood.entity_query_language.core.variable.Variable
used_by:
  - concept.krrood-eql
  - pycram.robot_plans.ActionDescription
  - pycram.querying.predicates
status: stable
tags: [krrood, eql, predicate, symbolic, boolean, condition, dual-mode]
last_ingest: 2026-05-17
---

_EQL boolean test: an ABC with a `__new__` dispatch that transparently returns an `InstantiatedVariable` when any constructor argument is a `Variable`, enabling the same concrete predicate class to work in both direct evaluation and symbolic expression composition._

## Purpose

Predicates are the boolean filters in EQL expressions. They enable action pre/post-condition
logic to be written once — using ordinary Python class definitions — and to work with both
concrete values (immediate evaluation) and symbolic `Variable` arguments (deferred composition).

The dual-mode trick is in `Predicate.__new__`: one class definition, two runtime roles.

## `@symbolic_function` decorator

```python
@symbolic_function
def get_end_effector_view(arm: Arms, robot: AbstractRobot) -> View:
    return robot.get_arm(arm).end_effector_view
```

`@symbolic_function` wraps any Python function. When called with any argument that is a
`Selectable` (the abstract base of `Variable`), the function is **not** called; instead an
`InstantiatedVariable` representing the deferred call is returned. When all arguments are
concrete, the original function executes normally.

This makes standard library functions composable with EQL without any EQL-awareness in the
function itself — no refactoring required.

## `Predicate.__new__` — dual-mode dispatch

```python
class GripperIsFree(Predicate):
    def __init__(self, manipulator: Manipulator):
        self.manipulator = manipulator
    def evaluate(self) -> bool:
        return self.manipulator.gripper_state == GripperState.OPEN
```

**With a concrete `Manipulator`:**
```python
pred = GripperIsFree(manipulator=robot.right_arm.manipulator)
# → normal GripperIsFree instance; pred.evaluate() → bool
```

**With a `Variable[Manipulator]`:**
```python
pred = GripperIsFree(manipulator=arm_variable)
# → InstantiatedVariable[GripperIsFree]; participates in and_() / or_() chains
```

`Predicate.__new__` inspects all constructor keyword arguments. If any value is a `Variable`,
it bypasses `__init__` entirely and returns `InstantiatedVariable(GripperIsFree, **kwargs)`.
This means concrete predicates never need to handle the symbolic case themselves.

## Concrete predicates

| Predicate | Logic |
|-----------|-------|
| `HasType(entity, types_)` | `isinstance(entity, types_)` — type check against a single type |
| `HasTypes(entity, type_list)` | `any(isinstance(entity, t) for t in type_list)` — any-of |

Both predicates benefit from `__new__` dual-mode dispatch: calling with a Variable argument
automatically yields a symbolic form.

## Related

**Uses:** [[krrood.entity_query_language.core.variable.Variable]]

**Used by:** [[concept.krrood-eql]], [[pycram.robot_plans.ActionDescription]]

## Open questions

- `Predicate.__new__` dispatch inspects **keyword** arguments. Positional-only Variable
  arguments would bypass dispatch — no runtime check prevents this usage pattern.
- Whether `HasType` / `HasTypes` are the only shipped concrete predicates, or whether
  pycram defines additional predicates elsewhere (e.g. `GripperIsFree`,
  `reachability_validator`) is not captured — the summary mentions them as examples in
  action pre-conditions, but their module location is unconfirmed.

## Provenance

- `krrood/src/krrood/entity_query_language/predicate.py:1-120` — `symbolic_function` decorator,
  `Predicate` ABC, `HasType`, `HasTypes`.
