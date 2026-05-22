---
id: krrood.entity_query_language.backends.QueryBackend
kind: entity
package: cross
source_paths:
  - path: krrood/src/krrood/entity_query_language/backends.py
    lines: [1, 246]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: krrood/src/krrood/entity_query_language/query/query.py
    lines: [1, 100]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - krrood.entity_query_language.core.variable.Variable
  - krrood.entity_query_language.query.Query
used_by:
  - pycram.datastructures.Context
  - pycram.plans.UnderspecifiedNode
  - concept.krrood-eql
status: stable
tags: [krrood, eql, backend, query, generative, selective, probabilistic, sql]
last_ingest: 2026-05-17
---

_Abstract EQL evaluation strategy plugged into [[pycram.datastructures.Context]] as `query_backend`; four implementations covering in-process Python, SQL, probabilistic sampling, and the default in-memory generative backend._

## Purpose

`QueryBackend` decouples EQL expression evaluation from its execution strategy. The same
`underspecified(NavigateAction)(target=variable(...))` expression evaluates in-memory via
`EntityQueryLanguageBackend`, against a database via `SQLAlchemyBackend`, or probabilistically
via `ProbabilisticBackend` — without any change to the expression or action code. The active
backend is stored on `Context.query_backend`.

## Interface

```python
class QueryBackend(ABC):
    def evaluate(self, expression: Query) -> Iterable[T]: ...
    def can_evaluate(self, expression: Query) -> bool: ...
```

## Backend hierarchy

```
QueryBackend
├── SelectiveBackend          — selects from existing domain objects
│   ├── ProbabilisticBackend  — samples from tractable probabilistic model
│   └── SQLAlchemyBackend     — translates EQL to SQL via eql_to_sql()
└── GenerativeBackend         — generates new T instances from variable domains
    └── EntityQueryLanguageBackend  — default in-process Python evaluation
```

## `SelectiveBackend`

`can_evaluate` → `True` for any `Query`. Operates over a fixed domain (world objects,
DB rows) and filters by `.where()` conditions. Does not construct new instances.

## `GenerativeBackend`

`can_evaluate` → `True` only for `Match[T]` expressions (generative query node requiring
a constructible type). Generates new `T` instances by grounding all `variable(...)` domains.

## `EntityQueryLanguageBackend` (default)

Plugged into `Context.query_backend` at robot startup.

`_evaluate_underspecified(match: Match[T])`:
1. Collects all `AttributeMatch` entries in the `Match` to identify which kwargs are Variables.
2. Calls `set_of(*variables).evaluate()` → yields Cartesian product of all variable domains.
3. For each binding combination, constructs `T(**grounded_kwargs)`.
4. Applies `.where()` filter conditions on the Match.
5. Yields the constructed `T` instance for each valid binding.

For queries that are not `Match`, falls through to `set_of(query).evaluate()` — no instance
construction, plain domain selection.

## `ProbabilisticBackend`

Uses `FullyFactorizedRegistry` — a tractable model treating all parameters as independent
random variables. `_sample_underspecified_parameters(Match[T])` draws parameter samples from
registered priors and yields grounded `T` instances. Non-deterministic between calls; suitable
when parameter distributions are known but exact target is uncertain.

## `SQLAlchemyBackend`

Translates EQL query structure to SQLAlchemy ORM queries via `eql_to_sql()`. Used when world
objects live in a relational DB (krrood/ormatic integration). Returns ORM-loaded objects.

## Related

**Uses:** [[krrood.entity_query_language.core.variable.Variable]], [[krrood.entity_query_language.query.Query]]

**Used by:** [[pycram.datastructures.Context]], [[pycram.plans.UnderspecifiedNode]], [[concept.krrood-eql]]

## Open questions

- `ProbabilisticBackend` uses `FullyFactorizedRegistry` — how priors are registered or learned
  for action parameters like `Pose` is not documented in the pycram codebase.
- Whether `EntityQueryLanguageBackend` handles nested `Match` (e.g. `underspecified(T)(x=underspecified(X)(...))`)
  is unverified — the grounding loop may not recurse into nested Match expressions.

## Provenance

- `krrood/src/krrood/entity_query_language/backends.py:1-246` — `QueryBackend`, `SelectiveBackend`,
  `GenerativeBackend`, `EntityQueryLanguageBackend`, `ProbabilisticBackend`, `SQLAlchemyBackend`.
- `krrood/src/krrood/entity_query_language/query/query.py:1-100` — `Query`, `SetOf`, `Entity`.
