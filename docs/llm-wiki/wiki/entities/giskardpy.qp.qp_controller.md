---
id: giskardpy.qp.qp_controller
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/qp_controller.py
    lines: [357, 465]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/qp/qp_controller_config.py
    lines: [1, 159]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/qp/solvers/qp_solver.py
    lines: [1, 36]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/executor.py
    lines: [80, 217]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.qp.constraint_collection.ConstraintCollection
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - giskardpy.qp.qp_data
  - giskardpy.qp.qp_data_factories
  - giskardpy.qp.adapters
  - giskardpy.motion_statechart.context
  - concept.qp-controller
used_by:
  - bridge.pycram-giskardpy
  - bridge.sdt-giskardpy
  - concept.qp-controller
  - giskardpy.motion_statechart.context
  - pycram.locations.locations.GiskardLocation
  - pycram.pose_validator
status: stable
tags: [giskardpy, qp, controller, solver, executor, config, mpc]
last_ingest: 2026-05-17
---

_Bundled page: `QPController`, `QPControllerConfig`, `QPSolver` (abstract + plugins), and `Executor`._

## `QPController`

```python
@dataclass
class QPController:
    config: QPControllerConfig
    degrees_of_freedom: InitVar[List[DegreeOfFreedom]]
    constraint_collection: ConstraintCollection
    world_state_symbols: List[FloatVariable]
    life_cycle_variables: List[FloatVariable]
    float_variables: List[FloatVariable]
```

`__post_init__` pipeline:
1. Creates `QPSolver` from `config.qp_solver_class()`.
2. Calls `QPDataSymbolic.from_giskard(active_dofs, constraint_collection, config)` to build a symbolic QP.
3. Gets the appropriate `QPDataFactory` subclass via `QPDataFactory.get_factory_from_qp_data_type(solver.qp_data_type)`.
4. Calls `factory.compile(world_state_symbols, life_cycle_symbols, float_variables)` — compiles CasADi functions and binds them to numpy memory views.

`_set_active_dofs` filters the DOF list to only those whose position/velocity/acceleration/jerk variables appear in the constraint expressions. This is the DOF filter — only variables that appear in at least one constraint are included.

**`compute_command(world_state, life_cycle_state, float_variables) → np.ndarray`**
Each tick:
1. `qp_data_factory.evaluate(…)` — evaluates compiled CasADi functions numerically
2. `qp_data_raw.apply_filters()` — removes zero-weight columns (inactive constraints)
3. `qp_solver.solver_call(qp_data_filtered)` → `xdot`
4. `xdot_to_control_commands(xdot)` — extracts velocity slice and divides by `mpc_dt²`

**`xdot_to_control_commands`** extracts `xdot[offset:offset+len(active_dofs)]` where `offset = len(active_dofs) * (prediction_horizon - 2)`. Divides by `mpc_dt²`. Assembles into `full_control_cmds` at `dof_filter` indices.

## `QPControllerConfig`

```python
@dataclass
class QPControllerConfig:
    target_frequency: float       # control loop Hz
    prediction_horizon: int = 7   # MPC steps; min 4
    max_derivative: Derivatives = Derivatives.jerk
    dof_weights: Dict[PrefixedName, DerivativeMap[float]] = ...  # default velocity=0.01
    retries_with_relaxed_constraints: int = 5
    qp_solver_class: Type[QPSolver] = QPSolverPIQP
```

`mpc_dt = control_dt = 1/target_frequency` is set in `__post_init__`. Warns if frequency < 20 Hz; raises if `prediction_horizon < 4`.

Tuning recipe:
1. Set `target_frequency` ≤ robot joint-state publish rate (simulation: 20 Hz).
2. Start with `prediction_horizon=7`.
3. Increase horizon if motion is not smooth; reduce Hz if QP solve can't keep up.

`QPControllerConfig.create_with_simulation_defaults()` → `target_frequency=20, prediction_horizon=7`.

## `QPSolver` — abstract base

```python
@dataclass
class QPSolver(Generic[T]):
    qp_data_type: Type[T]         # classmethod property
    def solver_call(qp_data: T) -> np.ndarray: ...
```

Standard QP form:
```
min_x  0.5 xᵀ H x + gᵀ x
s.t.   lb ≤ x ≤ ub        (box constraints)
       E x = bE             (equality)
       lbA ≤ A x ≤ ubA     (two-sided inequality)
```

Available implementations:

| Class | Backend | Notes |
|---|---|---|
| `QPSolverPIQP` | PIQP | Default; open-source interior-point |
| `QPSolverGurobi` | Gurobi | Commercial; supports IIS analysis for debugging infeasibility |
| `QPSolverQPALM` | QPALM | ADMM-based; good for large sparse problems |
| `QPSolverQPSWIFT` | qpSWIFT | Embedded systems focus |

## `Executor` — the control loop owner

```python
@dataclass
class Executor:
    context: MotionStatechartContext
    pacer: Pacer = SimulationPacer()
    trajectory_plotter: WorldStateTrajectoryPlotter | None = None
```

**`compile(msc)`**: compiles MSC, then `_compile_qp_controller()`. If no constraints exist, `qp_controller = None`.

**`tick()`**: increments cycle counter, optionally runs collision computation, calls `msc.tick()`, calls `qp_controller.compute_command()`, and calls `world.apply_control_commands(cmd, dt, max_derivative)`.

**`tick_until_end(timeout=1000)`**: runs the loop; in `finally` block zeros velocities/accelerations/jerks and calls `msc.cleanup_nodes()`. Raises `TimeoutError` if limit reached.

`Pacer` has two implementations:
- `SimulationPacer(real_time_factor=None)` — when `real_time_factor=None`, no sleep (run as fast as possible); when `1.0`, paces to match `control_dt`.

## Open questions

- `QPControllerDebugger.are_hard_limits_violated()` checks slack bounds after the solve. It's not clear from the source whether this runs automatically on every tick or only on exception paths.

## Related

- **Uses:** [[giskardpy.qp.constraint_collection.ConstraintCollection]], [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]], [[concept.qp-controller]], [[giskardpy.qp.qp_data]], [[giskardpy.qp.qp_data_factories]], [[giskardpy.qp.adapters]], [[giskardpy.motion_statechart.context]]
- **Used by:** [[bridge.pycram-giskardpy]], [[bridge.sdt-giskardpy]], [[giskardpy.motion_statechart.context]], [[pycram.locations.locations.GiskardLocation]], [[pycram.pose_validator]]

## Provenance

- `qp_controller.py:357-465` — `QPController` class.
- `qp_controller_config.py:1-159` — `QPControllerConfig` class.
- `qp/solvers/qp_solver.py:1-36` — `QPSolver` abstract base + QP form documentation.
- `executor.py:80-217` — `Executor`, `Pacer`, `SimulationPacer`.
