---
id: krrood.entity_query_language.query.Query
kind: entity
package: cross
source_paths:
  - path: krrood/src/krrood/entity_query_language/query/query.py
    lines: [1, 595]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - krrood.entity_query_language.core.variable.Variable
used_by:
  - krrood.entity_query_language.backends.QueryBackend
  - concept.krrood-eql
status: stable
tags: [krrood, eql, query, dag, fluent, projection, build-lock, cartesian-product]
last_ingest: 2026-05-17
---

_EQL query node: abstract Cartesian-product expression over `_selected_variables_` with a fluent build API (`where`/`having`/`grouped_by`/`ordered_by`/`distinct`/`limit`); once `build()` is called the DAG is locked against further structural modification._

## Purpose

`Query` is the composable, lazy query object in EQL. It wraps a set of selected variables
and a pipeline of filter/sort/group/limit operations, all assembled lazily into an expression
DAG. `QueryBackend.evaluate(query)` drives the actual iteration.

The build-lock constraint is central: once the DAG is wired, further structural edits raise
`TryingToModifyAnAlreadyBuiltQuery`. This prevents queries from being accidentally mutated
after being passed into a backend or composed into a parent expression.

## Class hierarchy

```
Query (ABC)
├── MultiArityExpressionThatPerformsACartesianProduct  — Cartesian product evaluation
└── CanBehaveLikeAVariable[T]                          — embeddable in other expressions
    ├── SetOf      — multi-variable result; wraps values in UnificationDict
    └── Entity[T]  — single-variable result; unwraps to direct value binding
```

## Key fields

| Field | Type | Notes |
|-------|------|-------|
| `_selected_variables_` | `Tuple[Selectable, ...]` | Variables projected into the output |
| `_built_` | `bool` | True after first `build()`; further structural edits raise |
| `_where_builder_` | `WhereBuilder \| None` | AND-chain of `.where()` conditions |
| `_having_builder_` | `HavingBuilder \| None` | Post-aggregation filter conditions |
| `_grouped_by_builder_` | `GroupedByBuilder \| None` | Group-by variables + aggregators |
| `_ordered_by_builder_` | `OrderedByBuilder \| None` | Sort key, direction, optional key function |
| `_quantifier_builder_` | `QuantifierBuilder` | Default `An` (yield all results) |
| `_results_mapping` | `List[ResultMapping]` | Post-evaluation transform pipeline (e.g. `distinct`) |

## `build()` — DAG wiring order

```python
def build(self) -> Self:  # idempotent; second call is a no-op except for ordered_by/quantifier re-apply
```

On first call, wires children in this evaluation order:
1. **having** (wraps grouped_by expression, requires grouped_by to exist)
2. **grouped_by** (if present but no having)
3. **where** (if present but no group)
4. **selected_variables** (Cartesian product of all selected vars — always present)
5. **ordered_by** wrapper (applied as `_expression_` override)
6. **quantifier** wrapper (`An` by default; `_expression_` ends up pointing here)

After build, `self._expression_` → topmost wrapper. `evaluate()` delegates to it.

## Fluent API

```python
SetOf(v1, v2)           # multi-variable query
  .where(pred_a, pred_b)  # AND-chain; multiple calls extend the condition list
  .having(agg_cond)       # post-group filter (requires grouped_by)
  .grouped_by(v1)         # group results; activates aggregation path
  .ordered_by(score_var, descending=True, key=lambda x: -x)
  .distinct(v1)           # deduplicate on v1; uses SeenSet for O(1) lookup
  .limit(10)              # cap at 10 results; raises NonPositiveLimitValue if n <= 0
```

`where()`, `having()`, `grouped_by()` carry `@modifies_query_structure` — they raise
`TryingToModifyAnAlreadyBuiltQuery` after `build()`. `limit()` and `ordered_by()` update
both the unbuilt spec and the live `_expression_` so they can be adjusted post-build.

## `evaluate()` entry point

```python
def evaluate(self) -> Iterator:
    self.build()
    return self._expression_.evaluate()  # delegates to quantifier-wrapped chain
```

## `SetOf` vs `Entity`

| Subclass | Output binding | Typical use |
|----------|---------------|-------------|
| `SetOf(v1, v2, ...)` | `{self._id_: UnificationDict({var: value})}` | multi-variable queries; inline EQL expressions |
| `Entity[T](v)` | `{self._id_: value}` | single-variable queries via `entity(v)` factory |

`SetOf`'s `UnificationDict` preserves `Variable → value` mapping so callers can extract
individual variable values from the result by identity, not by position.

## Context manager

```python
with SetOf(v).where(cond) as expr:
    # expr = _conditions_root_; pushed onto SymbolicExpression._symbolic_expression_stack_
```

Used in rule trees. `__enter__` forces `build()` first.

## `_invert_()` — unsupported

`Query._invert_()` raises `UnsupportedNegation(Query)`. Queries cannot appear inside `not_()`.

## Related

**Uses:** [[krrood.entity_query_language.core.variable.Variable]]

**Used by:** [[krrood.entity_query_language.backends.QueryBackend]], [[concept.krrood-eql]]

## Open questions

- The `_parent_` setter auto-builds the query when a non-Quantifier parent is assigned
  (tagged TODO in source). This means composing a query into a parent expression implicitly
  locks it — which can silently suppress `TryingToModifyAnAlreadyBuiltQuery` errors if the
  caller tries to add conditions afterwards.
- `CountAll` aggregator in `grouped_by` must be patched post-build to point at the
  `GroupedBy` expression — this is done in `_if_count_all_is_used_update_its_child...()`.
  Whether `CountAll` used outside of `grouped_by` contexts errors or silently produces
  incorrect results is unverified.

## Provenance

- `krrood/src/krrood/entity_query_language/query/query.py:1-595` — `Query`, `SetOf`, `Entity`,
  `@modifies_query_structure`, `_get_distinct_results_`, builder/quantifier delegation.
