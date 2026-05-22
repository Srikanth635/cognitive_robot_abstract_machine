---
id: sdt.spatial_computations.ik_solver
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_computations/ik_solver.py
    lines: [1, 608]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
  - sdt.world_description.world_entity.KinematicStructureEntity
  - concept.forward-kinematics
used_by:
  - sdt.world.World
  - sdt.spatial_types.spatial_types
status: stable
tags: [sdt, ik, inverse-kinematics, qp, daqp, jacobian, solver]
last_ingest: 2026-05-17
---

_`InverseKinematicsSolver`: QP-based IK that iteratively solves for joint positions bringing a tip frame to a target pose relative to a root frame._

## Purpose

Provides in-SDT IK separate from the giskardpy whole-body controller — used for things like computing reachability poses offline, computing robot base positions for grasp planning, and testing. The solver uses the `daqp` QP library directly (not giskardpy).

## Algorithm

The QP minimises:

```
min_{dof_v, slack}  0.001*||dof_v||²  +  2500*||slack||²
subject to
    J * dof_v * dt + slack = position_error (translation, 3)
    J * dof_v * dt + slack = rotation_error (rotation, 3)
    velocity_limits[i] ≤ dof_v[i] ≤ velocity_limits[i]
```

Slack variables absorb constraint violations; their high weight forces them toward zero. Each iteration: solve QP → integrate → check convergence → repeat.

**Convergence criteria:**
- `vel_below_threshold AND slack_below_threshold` → target reached.
- `vel_below_threshold AND slack_above_threshold` → `UnreachableException` — converged but not at target.
- Exceeds `max_iterations` → `MaxIterationsException`.

## Key classes

### `InverseKinematicsSolver`

```python
result: Dict[DegreeOfFreedom, float] = solver.solve(
    root, tip, target_htm, dt=0.05, max_iterations=200
)
```

### `QPProblem`

Constructed per-solve; extracts active/passive DOFs from the `root→tip` chain, sets up constraints and weights symbolically, compiles to CasADi functions. `evaluate_at_state(solver_state)` evaluates the compiled QP matrices (H, g, A, l, u) at the current joint positions.

DOF extraction uses `world.compute_split_chain_of_connections(root, tip)` to enumerate active and passive DOFs across the kinematic chain.

### `ConstraintBuilder`

Builds box constraints (velocity + position limits, with `sm.max`/`sm.min` clamping) and goal constraints (position + rotation errors with velocity caps). The Jacobian `J` is computed via `current_expr.jacobian(active_variables)` — a CasADi symbolic differentiation.

### `SolverState`

Mutable iteration state: `position: np.ndarray` (active DOF positions), `passive_position: np.ndarray`. `update_position(velocity, dt)` integrates velocity into position.

## Exceptions

| Exception | Condition |
|---|---|
| `UnreachableException` | Converged (velocity → 0) but residual slack > threshold |
| `MaxIterationsException` | Iteration limit reached without convergence |
| `QPSolverException` | `daqp` returned non-optimal exit flag |

`DAQPSolverExitFlag` enumerates all daqp exit codes.

## Related

- **Uses:** [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]], [[sdt.world_description.world_entity.KinematicStructureEntity]], [[concept.forward-kinematics]]
- **Used by:** [[sdt.world.World]] (via `compute_inverse_kinematics` method; also callable from `sdt.reasoning.predicates.reachable()`)

## Open questions

- The rotation error uses a small hack: `root_R_tip.dot(RotationMatrix.from_axis_angle(Vector3.Z(), -0.0001))` — a near-identity perturbation to avoid singularity in quaternion differentiation. The exact failure mode this prevents is not documented.

## Provenance

- `ik_solver.py:1-608` — `InverseKinematicsSolver`, `QPProblem`, `ConstraintBuilder`, `SolverState`, `QPMatrices`, `DAQPSolverExitFlag`, exception classes.
