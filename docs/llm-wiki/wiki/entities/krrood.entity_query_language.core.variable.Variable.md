---
id: krrood.entity_query_language.core.variable.Variable
kind: entity
package: cross
source_paths:
  - path: krrood/src/krrood/entity_query_language/core/variable.py
    lines: [1, 287]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - pycram.robot_plans.ActionDescription
  - concept.krrood-eql
  - krrood.entity_query_language.predicate.Predicate
  - krrood.entity_query_language.backends.QueryBackend
  - krrood.entity_query_language.query.Query
status: stable
tags: [krrood, eql, variable, symbolic, domain, lazy, re-enterable]
last_ingest: 2026-05-17
---

_Atomic EQL expression: a typed, lazy, re-enterable domain iterable that yields one `OperationResult` per domain value during symbolic evaluation — the primitive building block for all EQL queries._

## Purpose

`Variable[T]` is the foundational unit of krrood's Entity Query Language. It wraps a
domain iterable and yields one binding per domain element when evaluated. Composed with
logical operators (`and_`, `or_`), arithmetic operators, and predicates, it enables
declarative symbolic queries evaluated lazily at plan execution time.

Every concrete argument in an `ActionDescription` is wrapped in a singleton-domain
`Literal` by `bound_variables` — this lets action pre/post-conditions treat known values
symbolically for uniform composition with genuinely open Variables.

## Construction

```python
# via factory function (preferred):
v = variable(Pose, domain=CostmapLocation(...))

# directly:
v = Variable[Pose](_type_=Pose, _domain_=ReEnterableLazyIterable([pose1, pose2]))
```

`_update_domain_(new_domain)` wraps any plain iterable in `ReEnterableLazyIterable` — a
re-enterable wrapper that can be iterated multiple times by re-invoking the generator
factory. Re-entrability is required because EQL expressions may be evaluated more than once
(e.g. across multiple UnderspecifiedNode resolution attempts).

## Key attributes

| Name | Kind | Notes |
|------|------|-------|
| `_type_` | `type[T]` | Python type this variable represents; used by `InstantiatedVariable` to construct instances |
| `_id_` | UUID | Unique binding key; `OperationResult.bindings` maps `{_id_: value}` |
| `_name_` | `str` | Human-readable label; derived from `_type_.__name__` or `repr(first domain value)` |
| `_domain_` | `ReEnterableLazyIterable[T]` | Lazy iterable of candidate values |

## `_evaluate__()` — core evaluation protocol

```python
def _evaluate__(self) -> Iterable[OperationResult]:
    for v in self._domain_:
        yield OperationResult(bindings={self._id_: v})
```

Each yielded `OperationResult` carries the binding `{self._id_: value}` for one domain
element. Parent expressions (AND, OR, Cartesian product) merge bindings from child results
to build a combined binding set for each valid combination.

## Subclasses

### `Literal[T]`

```python
Literal[T](value: T)
```

Singleton-domain variable: `_domain_ = [value]`. Has a `.value` property returning the
single element. Used by `ActionDescription.bound_variables` to wrap each known field value
in a single-element domain for uniform predicate composition.

### `InstantiatedVariable[T]`

```python
InstantiatedVariable[T](_type_, *child_variables)
```

Subclasses `MultiArityExpressionThatPerformsACartesianProduct`. Iterates all Cartesian
product combinations of its child variables' domains and for each combination constructs
`_type_(**grounded_kwargs)` — one new instance of `T` per binding. This is the *generative*
mode of EQL: `underspecified(NavigateAction)` returns an `InstantiatedVariable[NavigateAction]`.

`_evaluate_product_` drives the Cartesian iteration; each child variable provides its
binding, and the combined binding constructs one `T` instance.

### `ExternallySetVariable[T]`

```python
ExternallySetVariable[T](_type_)
```

Empty domain — yields nothing on evaluation. Value is injected by external code before
evaluation runs. Used for context-provided values (e.g. active ROS node handle) that are
not enumerated but supplied programmatically.

## Related

**Used by:** [[pycram.robot_plans.ActionDescription]], [[concept.krrood-eql]],
[[krrood.entity_query_language.predicate.Predicate]], [[krrood.entity_query_language.backends.QueryBackend]]

**See also:** [[concept.krrood-eql]]

## Open questions

- `InstantiatedVariable` calls `_type_(**kwargs)` with keyword arguments built from child
  variable bindings. If any required constructor argument is missing from the combined
  bindings, the construction raises — no error-handling path for partial bindings exists.

## Provenance

- `krrood/src/krrood/entity_query_language/core/variable.py:1-287` — `Variable`, `Literal`,
  `InstantiatedVariable`, `ExternallySetVariable`, `ReEnterableLazyIterable`.
