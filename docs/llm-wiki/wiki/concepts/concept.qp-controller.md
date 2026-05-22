---
id: concept.qp-controller
kind: concept
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/qp_controller.py
    lines: [357, 465]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/qp/constraint_collection.py
    lines: [1, 60]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/executor.py
    lines: [80, 217]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.qp.constraint
  - giskardpy.qp.constraint_collection.ConstraintCollection
  - giskardpy.qp.qp_controller
  - concept.motion-statechart
used_by:
  - giskardpy.qp.qp_controller
  - giskardpy.qp.constraint_collection.ConstraintCollection
  - bridge.pycram-giskardpy
status: stable
tags: [concept, giskardpy, qp, controller, constraint, solver, mpc]
last_ingest: 2026-05-17
---

_The giskardpy QP layer: from symbolic constraint collection through MPC formulation to per-cycle joint velocity commands._

## Overview

Every `Task.build()` produces a `ConstraintCollection` — a set of symbolic CasADi constraints on robot state. After the MSC is compiled, these collections are merged and handed to the `QPController`. Each tick, the controller evaluates the merged constraints numerically and calls a QP solver to produce the next joint velocity command.

## Architecture

```
MotionStatechart
  ├── Task A → ConstraintCollection A
  ├── Task B → ConstraintCollection B
  └── ...
        ↓  combine_constraint_collections_of_nodes()
    ConstraintCollection (merged, lifecycle-gated)
        ↓
    QPDataFactory.compile() → QPDataFactory.evaluate() each tick
        ↓
    QPSolver.solver_call(qp_data) → xdot
        ↓
    QPController.xdot_to_control_commands() → control_cmds
        ↓
    World.apply_control_commands()
```

## Lifecycle gating

`ConstraintCollection.link_to_motion_statechart_node(node)` multiplies every constraint's `quadratic_weight` by `is_running = if_eq(lifecycle_var, RUNNING, 1, 0)`. This means:
- Constraints from non-RUNNING nodes have zero weight → the QP solver effectively ignores them.
- No filtering or rebuilding is needed each tick — the QP matrix dimensions stay fixed; weights change numerically.

This is called automatically from `MotionStatechart._build_and_apply_artifacts()` during `compile()`.

## MPC formulation

`QPControllerConfig` controls the formulation:

| Parameter | Default | Role |
|---|---|---|
| `target_frequency` | required | Control loop Hz; `mpc_dt = 1/target_frequency` |
| `prediction_horizon` | 7 | Steps in receding horizon; min 4 (jerk integration) |
| `max_derivative` | `jerk` | Highest derivative in decision variables |
| `dof_weights` | `velocity=0.01` per DOF | Per-DOF velocity cost (soft regularization) |
| `retries_with_relaxed_constraints` | 5 | Infeasibility recovery by relaxing slack bounds |

The QP decision variable `xdot` has shape `len(active_dofs) × prediction_horizon` (approximately). The control command is extracted from the slice at `offset = len(active_dofs) × (prediction_horizon - 2)` and divided by `mpc_dt²`.

## Two constraint families

| Family | Base class | Semantics | Normalization |
|---|---|---|---|
| Integral (position-level) | `IntegralConstraint` | `expression * control_horizon + slack ≈ bound`; error capped to `normalization_factor × dt × horizon` | `weight / (normalization_factor² × horizon)` |
| Derivative (velocity/acc/jerk) | `DerivativeConstraint` | `derivative(expression)` within `[lower_limit, upper_limit]` | `weight / normalization_factor²` |

Position-level constraints are cheaper: one constraint per DOF vs. one per step per DOF.

## QP solver plugins

`QPControllerConfig.qp_solver_class` (default `QPSolverPIQP`) is swappable. Available implementations: `QPSolverPIQP`, `QPSolverGurobi`, `QPSolverQPALM`, `QPSolverQPSWIFT`. The standard problem form is:

```
min_x  0.5 xᵀ H x + gᵀ x
s.t.   lb ≤ x ≤ ub
       E x = bE
       lbA ≤ A x ≤ ubA
```

## Execution loop (`Executor`)

`Executor.tick_until_end(timeout=1000)` runs the full loop:
```python
for _ in range(timeout):
    msc.tick(context)            # update observation + lifecycle
    cmd = qp.compute_command(…)  # evaluate QP, extract velocities
    world.apply_control_commands(cmd, dt, max_derivative)
    pacer.sleep()                # real-time pacing
    if msc.is_end_motion(): break
finally:
    world.state.velocities[:] = 0
    msc.cleanup_nodes(context)
```

If the MSC has no constraints (e.g., only monitoring nodes), `qp_controller` is `None` and only the MSC tick runs.

## Related

- **Uses:** [[giskardpy.qp.constraint]], [[giskardpy.qp.constraint_collection.ConstraintCollection]], [[giskardpy.qp.qp_controller]], [[concept.motion-statechart]]
- **Used by:** [[bridge.pycram-giskardpy]] (via `MotionExecutor`)

## Provenance

- `qp_controller.py:357-465` — `QPController.__post_init__`, `compute_command`, `xdot_to_control_commands`.
- `constraint_collection.py:113-120` — `link_to_motion_statechart_node` lifecycle gating.
- `executor.py:80-217` — `Executor`, `tick_until_end`, `_compile_qp_controller`.
