---
id: giskardpy.qp.constraint
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/constraint.py
    lines: [1, 132]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - giskardpy.qp.constraint_collection.ConstraintCollection
  - concept.qp-controller
status: stable
tags: [giskardpy, qp, constraint, slack, integral, derivative, equality, inequality]
last_ingest: 2026-05-17
---

_Bundled page: the full constraint class hierarchy — `BaseConstraint`, two integral families (position-level), and two derivative families (velocity/acc/jerk-level)._

## `BaseConstraint`

```python
@dataclass
class BaseConstraint:
    name: str
    expression: Scalar          # CasADi symbolic expression (must be shape (1,1))
    quadratic_weight: ScalarData
    linear_weight: ScalarData
```

All other constraint types extend this. `name` must be unique within a `ConstraintCollection`; duplicates raise `DuplicateNameException`.

## Integral constraint family (position-level)

`IntegralConstraint(BaseConstraint)` adds:
- `normalization_factor: float` — the expected rate of change of `expression` (e.g. joint velocity limit in m/s or rad/s for position expressions). Used to scale the constraint so different units are comparable.
- `normalized_weight(control_horizon)` → `weight / (normalization_factor² × control_horizon)`
- `_apply_cap(value, dt, control_horizon)` → `clamp(value, -normalization_factor × dt × horizon, +…)` — limits error magnitude to what can be achieved in one horizon.

### `EqualityConstraint(IntegralConstraint)`

Adds: `bound`, `lower_slack_limit`, `upper_slack_limit`.

QP row: `expression * control_horizon + slack = capped_bound(dt, horizon)`.

### `InequalityConstraint(IntegralConstraint)`

Adds: `lower_error`, `upper_error`, `lower_slack_limit`, `upper_slack_limit`.

QP row: `capped_lower_error ≤ expression * control_horizon + slack ≤ capped_upper_error`.

## Derivative constraint family (velocity/acc/jerk-level)

`DerivativeConstraint(BaseConstraint)` adds:
- `derivative: Derivatives` — which derivative to constrain (`velocity`, `acceleration`, `jerk`)
- `normalization_factor: ScalarData` — for normalization, same unit as `derivative(expression)`
- `normalized_weight()` → `weight × (1/normalization_factor)²`

Unlike `IntegralConstraint`, there is no `control_horizon` factor because derivative constraints apply directly (one constraint per step), not as an integral over the horizon.

### `DerivativeInequalityConstraint(DerivativeConstraint)`

Adds: `lower_limit`, `upper_limit`, `lower_slack_limit`, `upper_slack_limit`.

Used by `add_velocity_constraint` to keep joint velocities in `[lower_limit, upper_limit]`.

### `DerivativeEqualityConstraint(DerivativeConstraint)`

Adds: `bound`, `lower_slack_limit`, `upper_slack_limit`.

Used by `add_velocity_eq_constraint` to set a target velocity (e.g. diff-drive wheel velocity).

## Hierarchy summary

```
BaseConstraint
├── IntegralConstraint
│   ├── EqualityConstraint          ← position task goals
│   └── InequalityConstraint        ← position range goals
└── DerivativeConstraint
    ├── DerivativeEqualityConstraint  ← velocity setpoints
    └── DerivativeInequalityConstraint ← velocity limits
```

## Related

- **Used by:** [[giskardpy.qp.constraint_collection.ConstraintCollection]], [[concept.qp-controller]]

## Provenance

- `constraint.py:1-132` — all six dataclasses.
