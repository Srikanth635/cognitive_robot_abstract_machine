---
id: giskardpy.qp.qp_data
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/qp_data.py
    lines: [1, 459]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - giskardpy.qp.qp_data_factories
  - giskardpy.qp.qp_controller
status: stable
tags: [giskardpy, qp, data, numpy, sparse, format, solver]
last_ingest: 2026-05-17
---

_Bundled page: `QPData` (ABC), `QPDataExplicit` (canonical 8-array format), and `QPDataTwoSidedInequality` (flat format required by some solvers)._

## `QPData` — abstract base

```python
@dataclass
class QPData(ABC):
    num_equality_slack_variables: int
    num_inequality_slack_variables: int

    @property
    def num_slack_variables(self) -> int: ...

    @abstractmethod
    def apply_filters(self) -> Self: ...
```

`apply_filters()` removes optimization variables with zero quadratic weight — inactive constraints. **DOF variables are never removed** (the filter forces `True` for all non-slack indices). Only slack variables whose parent constraint has zero weight are dropped.

---

## `QPDataExplicit`

The canonical expanded format. Eight numpy arrays + two slack counts:

```
min_x  0.5 xᵀ diag(quadratic_weights) x + linear_weights^T x
s.t.   box_lower_constraints ≤ x ≤ box_upper_constraints
       equality_matrix @ x  == equality_bounds
       inequality_lower_bounds ≤ inequality_matrix @ x ≤ inequality_upper_bounds
```

| Field | Type | Notes |
|---|---|---|
| `quadratic_weights` | `np.ndarray` | Diagonal of Hessian |
| `linear_weights` | `np.ndarray` | Linear objective term |
| `box_lower_constraints` | `np.ndarray` | Per-variable lower bounds |
| `box_upper_constraints` | `np.ndarray` | Per-variable upper bounds |
| `equality_matrix` | `sp.csc_matrix` | E (sparse CSC) |
| `equality_bounds` | `np.ndarray` | b_E |
| `inequality_matrix` | `sp.csc_matrix` | A (sparse CSC) |
| `inequality_lower_bounds` | `np.ndarray` | lb_A |
| `inequality_upper_bounds` | `np.ndarray` | ub_A |

### `apply_filters()` logic

```python
zero_quadratic_weight_filter = quadratic_weights != 0
zero_quadratic_weight_filter[:-num_slack_variables] = True  # never filter DOFs
```

The slack sub-filter is split:
- `bE_filter` — rows to keep in equality matrix (slack with non-zero weight)
- `bA_filter` — rows to keep in inequality matrix

Then columns (optimization variables) and rows (constraint rows) are sliced together from the sparse matrices.

### `to_two_sided_inequality()` conversion

Stacks `[I; E; A]` into a single `inequality_matrix`, concatenates all bounds.
Used when the solver requires the unified format (e.g., QPALM, PIQP).

### Debug utilities

- `pretty_print_problem()` → human-readable string with numpy + scipy constructors (used to create test cases for infeasible QPs)
- `analyze_well_posedness()` → condition number analysis of H and A/E matrices; warns if condition number > 1000

---

## `QPDataTwoSidedInequality`

The flat format where box constraints and equality constraints are merged into the
inequality matrix:

```
min_x  0.5 xᵀ diag(quadratic_weights) x + linear_weights^T x
s.t.   inequality_lower_bounds ≤ inequality_matrix @ x ≤ inequality_upper_bounds
```

The combined `inequality_matrix` has layout `[I_box; E; A]` from `to_two_sided_inequality()`.
Lazy accessor properties extract logical sub-regions without copying:

| Property | Returns |
|---|---|
| `box_lower_constraints` | `inequality_lower_bounds[:num_box_constraints]` |
| `box_upper_constraints` | `inequality_upper_bounds[:num_box_constraints]` |
| `eq_matrix` | `inequality_matrix[start_eq:start_neq, :]` |

The `_direct_limit_model` / `_cached_eyes` pair caches the identity sub-matrix via `@lru_cache` — creating identity matrices for constraint filtering is expensive; caching saves repeat allocation across ticks.

### `apply_filters()` semantics (TwoSided variant)

The TwoSided filter combines `zero_quadratic_weight_filter` (columns) with
`box_finite_filter` (rows where at least one bound is finite) before pruning. This is
slightly different from the Explicit variant — infinite box bounds are additionally dropped.

---

## Relationship to QP pipeline

```
ConstraintCollection
  → QPDataSymbolic.from_giskard()   [symbolic expressions]
  → QPDataFactory.compile()         [compiled CasADi functions]
  → QPDataFactory.evaluate()        → QPDataExplicit / QPDataTwoSidedInequality
  → QPData.apply_filters()          [remove zero-weight columns]
  → QPSolver.solver_call()          → xdot
```

`QPDataExplicit` is the intermediate format; `QPDataTwoSidedInequality` is the solver-facing
format for solvers that require it (produced either by `to_two_sided_inequality()` or directly
by `QPDataTwoSidedInequalityFactory`).

## Related

- Assembled by: [[giskardpy.qp.adapters]]
- Produced by: [[giskardpy.qp.qp_data_factories]]
- Consumed by: [[giskardpy.qp.qp_controller]]
- Constraint source: [[giskardpy.qp.constraint_collection.ConstraintCollection]]

## Provenance

- `giskardpy/src/giskardpy/qp/qp_data.py:1-459` at commit `0528d8cf3` — complete file.
