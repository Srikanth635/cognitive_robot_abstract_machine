---
id: sdt.world_description.world_state.WorldState
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_state.py
    lines: [29, 392]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
used_by:
  - sdt.world.World
  - sdt.spatial_computations.forward_kinematics
  - concept.forward-kinematics
status: stable
tags: [sdt, state, dof, numpy, thread-safe, mutablemapping, version]
last_ingest: 2026-05-17
---

_`MutableMapping[UUID, WorldStateEntryView]` backed by a `4×N float64` numpy array; the single ground-truth state bus for all DOF position/velocity/acceleration/jerk values in the world._

## Purpose

`WorldState` is the shared runtime state that both SDT's FK evaluation and giskardpy's QP solver read and write. Its `4×N` layout (rows = derivatives, columns = DOFs) lets the FK compiled functions bind `state.positions` as a zero-copy memory view. A monotonically increasing `version` counter lets consumers (FK manager, RayTracer) detect stale caches without notification callbacks.

## Layout

```
_data: np.ndarray  # shape (4, N), float64
        row 0: positions        ← FK evaluation input
        row 1: velocities       ← QP output / drive integration
        row 2: accelerations
        row 3: jerks
```

`positions`, `velocities`, `accelerations`, `jerks` are properties that return the respective row slice via `get_derivative(Derivatives.*)`.

## Key operations

| Operation | Method | Notes |
|---|---|---|
| Add DOF | `add_degree_of_freedom(dof)` | Calls `dof.create_variables()`, appends column, sets initial position within limits |
| Read | `state[dof_id].position` | Returns `WorldStateEntryView` (view into column) |
| Write | `state[dof_id].position = v` | Writes via `WorldStateEntryView.__setitem__` |
| Apply control | `_apply_control_commands(cmds, dt, derivative)` | Writes derivative level, integrates down to position |
| Merge | `merge_state(other)` | Overwrites all derivatives for matching DOFs |

## `WorldStateEntryView`

Returned by `state[dof_id]`. Wraps a 1D view into `_data[:, col]` with `position`, `velocity`, `acceleration`, `jerk` properties. All reads/writes hold `world._world_lock`.

## Thread safety

All operations acquire `world._world_lock` (a `threading.RLock`). `WorldStateEntryView` holds the same lock for its per-property access.

## `version` counter

Incremented by `_notify_state_change()` which also fires `StateChangeCallback` hooks. Callers comparing `version` snapshots can tell whether state has changed since their last read — the `RayTracer` uses `world.state.version` for this purpose.

## Related

- **Uses:** [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] (column added via `add_degree_of_freedom`; `create_variables()` called here)
- **Used by:** [[sdt.world.World]] (owns `state`), [[sdt.spatial_computations.forward_kinematics]] (binds `positions` as memory view), [[concept.forward-kinematics]]

## Open questions

- `_apply_control_commands` integrates only downward (velocity → position). There is no upward integration. This means if giskardpy writes velocities, positions are updated; but if giskardpy writes jerks, only jerk is updated — lower derivatives remain stale until the next integration pass.

## Provenance

- `world_state.py:29-392` — `WorldStateEntryView`, `WorldState`, `_add_dof`, `add_degree_of_freedom`, `_apply_control_commands`, `merge_state`, `get/set_derivative`, `position_float_variables`.
