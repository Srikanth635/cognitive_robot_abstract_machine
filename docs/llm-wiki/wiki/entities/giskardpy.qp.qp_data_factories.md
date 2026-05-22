---
id: giskardpy.qp.qp_data_factories
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/qp_data_factories.py
    lines: [1, 329]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.qp.qp_data
  - giskardpy.qp.adapters
used_by:
  - giskardpy.qp.qp_controller
status: stable
tags: [giskardpy, qp, factory, compile, casadi, numpy, sparse]
last_ingest: 2026-05-17
---

_Bundled page: `QPDataFactory[T]` (generic abstract), `QPDataExplicitFactory`, `QPDataTwoSidedInequalityFactory` — compile symbolic QP expressions to callable CasADi functions, then evaluate them numerically each tick._

## `QPDataFactory[T]` — abstract base

```python
@dataclass
class QPDataFactory(Generic[T], ABC):
    qp_data: QPDataSymbolic

    @classmethod
    @property
    def qp_data_type(cls) -> type[T]: ...          # resolved via Generic typevar

    @classmethod
    def get_factory_from_qp_data_type(cls, qp_data_type) -> type[QPDataFactory]: ...

    @abstractmethod
    def compile(self, world_state_symbols, life_cycle_symbols, float_variables): ...

    @abstractmethod
    def evaluate(self, world_state, life_cycle_state, float_variables) -> T: ...
```

`get_factory_from_qp_data_type` walks `cls.__subclasses__()` and returns the first whose
`qp_data_type` matches — used by `QPController.__post_init__` to pick the right factory
automatically. Adding a new solver format requires only subclassing `QPDataFactory[NewFormat]`.

---

## `QPDataExplicitFactory`

Produces [[giskardpy.qp.qp_data|QPDataExplicit]] each tick.

### `compile()` — two matrix functions + one combined vector function

```python
# Equality matrix (sparse CasADi → scipy.csc)
eq_matrix = hstack([eq_matrix_dofs, eq_matrix_slack, zeros_neq_slack])
self.equality_matrix_compiled = eq_matrix.compile(parameters=..., sparse=True)

# Inequality matrix (sparse)
neq_matrix = hstack([neq_matrix_dofs, zeros_eq_slack, neq_matrix_slack])
self.inequality_matrix_compiled = neq_matrix.compile(parameters=..., sparse=True)

# All 7 vector outputs in ONE compiled function (zero-copy via views)
self.combined_vector_f = CompiledFunctionWithViews(
    expressions=[
        qp_data.quadratic_weights,
        qp_data.linear_weights,
        qp_data.box_lower_constraints,
        qp_data.box_upper_constraints,
        qp_data.eq_bounds,
        qp_data.neq_lower_bounds,
        qp_data.neq_upper_bounds,
    ],
    parameters=...,
)
```

`CompiledFunctionWithViews` concatenates all expressions into a single CasADi output vector
and returns numpy memory views into slices — one function call produces 7 arrays with no copy.

### `evaluate()` — three function calls per tick

```python
eq_matrix_np   = equality_matrix_compiled(*args)      # sparse
neq_matrix_np  = inequality_matrix_compiled(*args)    # sparse
(qw, lw, lb, ub, be, lba, uba) = combined_vector_f(*args)  # 7 dense views
return QPDataExplicit(...)
```

---

## `QPDataTwoSidedInequalityFactory`

Produces [[giskardpy.qp.qp_data|QPDataTwoSidedInequality]] directly (avoids post-hoc conversion from Explicit).

### `compile()` — single matrix + combined vector

If no inequality constraints exist, only the equality matrix is used. Otherwise
eq and neq are vstacked into a single `constraint_matrix`:

```python
constraint_matrix = vstack([
    hstack([eq_dofs, eq_slack, zeros_neq]),
    hstack([neq_dofs, zeros_eq, neq_slack]),
])
self.inequality_matrix_compiled = constraint_matrix.compile(..., sparse=True)
```

The `combined_vector_f` has `additional_views` parameter — two extra view slices are
registered to directly extract `inequality_lower_bounds` and `inequality_upper_bounds`
as views from the concatenated output (slices defined by pre-computed lengths, zero-copy).

### `evaluate()` — two function calls per tick

```python
neq_matrix = inequality_matrix_compiled(*args)
(qw, lb_box, _, _, ub_box, _, _, lw,
 box_eq_neq_lower, box_eq_neq_upper) = combined_vector_f(*args)
return QPDataTwoSidedInequality(...)
```

The `_` slots correspond to internal view segments that are not extracted individually
(eq_bounds appears twice in expressions to serve both lb and ub via the two-sided format).

---

## Position in the QP pipeline

```
QPDataSymbolic (symbolic) ──compile()──► QPDataFactory
                                              │
                              evaluate() each tick
                                              │
                                    QPDataExplicit / QPDataTwoSidedInequality
                                              │
                                    apply_filters()
                                              │
                                    QPSolver.solver_call()
```

## Open questions

- `QPDataTwoSidedInequalityFactory.evaluate` stores the result in `self.qp_data_raw` (an instance attribute). This is unusual — factories are typically stateless. It may be a debug handle for the last evaluated QP. Needs verification whether this is intentional or accidental state.

## Related

- Symbolic input: [[giskardpy.qp.adapters]] (`QPDataSymbolic`)
- Output types: [[giskardpy.qp.qp_data]]
- Used by: [[giskardpy.qp.qp_controller]]

## Provenance

- `giskardpy/src/giskardpy/qp/qp_data_factories.py:1-329` at commit `0528d8cf3` — complete file.
