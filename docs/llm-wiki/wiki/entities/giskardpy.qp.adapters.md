---
id: giskardpy.qp.adapters
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/adapters/qp_adapter.py
    lines: [1, 1689]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.qp.constraint_collection.ConstraintCollection
  - giskardpy.qp.pos_in_vel_limits
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
  - sdt.spatial_types.Pose
used_by:
  - giskardpy.qp.qp_controller
  - giskardpy.qp.qp_data_factories
  - bridge.sdt-giskardpy
status: stable
tags: [giskardpy, qp, symbolic, adapter, mpc, casadi, dof, weights, bounds]
last_ingest: 2026-05-17
---

_`QPDataSymbolic` assembles the full symbolic QP problem (H, g, lb, ub, E, bE, A, lbA, ubA) from `DegreeOfFreedom` objects and a `ConstraintCollection` by delegating to six `ProblemDataPart` helpers. This is the conversion layer between SDT's DOF world model and giskardpy's QP matrices._

## Architecture

```
ConstraintCollection  +  List[DegreeOfFreedom]  +  QPControllerConfig
          │
          └─► QPDataSymbolic.from_giskard()
                │
                ├── Weights.construct_expression()         → (quadratic_w, linear_w)
                ├── FreeVariableBounds.construct_expression() → (lb, ub)  [box constraints]
                ├── EqualityModel.construct_expression()   → (E_dof, E_slack)
                ├── EqualityBounds.construct_expression()  → bE
                ├── InequalityModel.construct_expression() → (A_dof, A_slack)
                └── InequalityBounds.construct_expression() → (lbA, ubA)
```

`QPDataSymbolic` holds all six sub-parts plus their symbolic outputs. These are passed to `QPDataFactory.compile()` which compiles them into callable CasADi functions.

## `ProblemDataPart` — base for all six helpers

```python
@dataclass(eq=False)
class ProblemDataPart(ABC):
    degrees_of_freedom: List[DegreeOfFreedom]
    constraint_collection: ConstraintCollection
    config: QPControllerConfig

    control_horizon = prediction_horizon - max_derivative + 1
```

Each part knows the full DOF list, all constraints, and config. They all implement
`construct_expression()` returning a symbolic CasADi vector or matrix.

### MPC variable layout

The QP optimization variable `x` has layout over the prediction horizon:

```
[vel_dof1_t0, vel_dof2_t0, ..., vel_dofN_t0,
 vel_dof1_t1, ..., vel_dofN_t(ph-3),         ← (ph-2) velocity steps (excludes last 2)
 jerk_dof1_t0, ..., jerk_dofN_t(ph-1),       ← ph jerk steps
 eq_slack_1, ..., eq_slack_K,
 neq_slack_1, ..., neq_slack_M]
```

Acceleration is eliminated (expressed via velocity + jerk via the `no_acc` derivative link model).

## `Weights`

Objective weights are normalized per DOF limit and grow linearly over the horizon
(`horizon_weight_gain_scalar`). Formula:

```
weight(t) = base_weight * (1/limit)² * linear_ramp(t, alpha)
```

DOF acceleration weights are skipped (acceleration is implicit). Slack variable weights
come from `constraint.normalized_weight(control_horizon)`.

## `FreeVariableBounds` (box constraints)

Calls `velocity_limit(dof, max_derivative)` for each DOF, which runs `b_profile()`:
- MPC-optimal velocity profile given pos/vel/acc/jerk limits (via `simple_mpc`).
- If the position range is smaller than `vel_limit * dt`, velocity is clamped.
- When near a position limit, the profile is shifted to enforce stopping.

`b_profile()` is `@memoize` — expensive; called once per DOF, result reused.

## `EqualityModel` / `EqualityBounds`

`EqualityModel` builds the derivative-link rows (the "no-acc" formulation:
`vt = vt-1 + 2*vt-2 - jt * dt²`) plus equality constraint Jacobians tiled across
the control horizon.

`EqualityBounds` provides the RHS of those equalities.

## `InequalityModel` / `InequalityBounds`

Builds the implicit jerk-limit rows (inequality form of the position-limit profile)
and inequality constraint Jacobians tiled across the control horizon.

## `ForwardKinematicsBinding`

In `binding_policy.py` (in this package):

```python
@dataclass
class ForwardKinematicsBinding:
    float_variable_data: FloatVariableData
    name: PrefixedName
    root: KinematicStructureEntity   # SDT type
    tip: KinematicStructureEntity    # SDT type
    ...

    def bind(self, world: World):
        root_T_tip = world.compute_forward_kinematics_np(root, tip)
        float_variable_data.set_value(root_T_tip_expr, root_T_tip[:3,:4].T.flatten())
```

The 3×4 submatrix (rotation + translation) is stored as 12 auxiliary `FloatVariable` entries.
Constraint expressions reference these variables; `bind()` is called from `on_start()` or
each tick depending on `GoalBindingPolicy` (`Bind_at_build` / `Bind_on_start`).

This is how SDT FK results enter the CasADi expression graph without direct CasADi FK
computation — the numerical FK is re-evaluated each tick and injected as parameters.

## Key observations

- `_sorter()` deterministically sorts all dict-keyed expressions before stacking — critical
  for reproducibility across builds (ensures column ordering is stable).
- `find_best_jerk_limit()` runs a 100-iteration binary search QP to find the jerk limit that
  achieves a target velocity at the given horizon and dt. This happens at compile time, cached.
- `DegreeOfFreedom.has_position_limits()` gates whether position-limit inequality rows are added.
  Continuous revolute joints (unlimited rotation) skip these rows entirely.

## Related

- **DOF source:** [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]]
- **Constraint source:** [[giskardpy.qp.constraint_collection.ConstraintCollection]]
- **Output format:** [[giskardpy.qp.qp_data]] (`QPDataSymbolic` → compiled → `QPDataExplicit`)
- **Factory (compiles output):** [[giskardpy.qp.qp_data_factories]]
- **SDT bridge:** [[bridge.sdt-giskardpy]]

## Open questions

- The `no_acc` derivative link formulation (eliminating acceleration) appears to differ from
  the original giskardpy model (which had explicit acceleration rows). Whether this formulation
  reduces QP size or causes numerical issues with high-bandwidth motions is undocumented.

## Provenance

- `giskardpy/src/giskardpy/qp/adapters/qp_adapter.py:1-1689` at commit `0528d8cf3` — complete file (`QPDataSymbolic`, `ProblemDataPart`, `Weights`, `FreeVariableBounds`, `EqualityModel`, `EqualityBounds`, `InequalityModel`, `InequalityBounds`).
- `giskardpy/src/giskardpy/motion_statechart/binding_policy.py:1-81` at commit `0528d8cf3` — `ForwardKinematicsBinding`, `GoalBindingPolicy`.
