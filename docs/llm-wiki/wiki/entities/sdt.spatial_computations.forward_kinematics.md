---
id: sdt.spatial_computations.forward_kinematics
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_computations/forward_kinematics.py
    lines: [1, 196]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.Connection
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.world_state.WorldState
  - concept.forward-kinematics
used_by:
  - concept.forward-kinematics
  - sdt.world.World
  - sdt.spatial_types.spatial_types
status: stable
tags: [sdt, fk, kinematics, casadi, batch, compiled, memoize, numpy]
last_ingest: 2026-05-17
---

_`ForwardKinematicsManager`: compiles and evaluates batched FK expressions for all kinematic nodes; zero-copy tick evaluation via `CompiledFunction` bound to `WorldState._data`._

## Purpose

This is the computation engine behind every `world.compute_forward_kinematics(root, tip)` call. It maintains a BFS-order cache of symbolic root→node FK expressions, compiles them into a single CasADi function, and evaluates the whole batch in one call each tick.

## Key lifecycle

1. **Model change → `_notify()`**: re-runs `update_root_T_kse_expression_cache()` + `compile()` + `recompute()`.
2. **State change → `recompute()`**: clears memoization, calls `compiled_all_fks.evaluate()`.
3. **Query → `compute_np(root, tip)`**: index into `forward_kinematics_for_all_bodies` via UUID offset map; result is `@memoize`-cached within one tick.

## Key attributes

| Attribute | Type | Notes |
|---|---|---|
| `compiled_all_fks` | `CompiledFunction` | Bound to `world.state.positions` memory view |
| `forward_kinematics_for_all_bodies` | `np.ndarray` | `4N×4` batch result; 4 rows per body |
| `body_id_to_all_fk_index` | `Dict[UUID, int]` | UUID → row offset in the batch array |
| `root_T_kse_expression_cache` | `Dict[UUID, HTM]` | Symbolic root→node FK per node ID |

## Key methods

| Method | Returns | Decorator |
|---|---|---|
| `compose_expression(root, tip)` | `HTM` (symbolic) | `@copy_memoize` — deep copy of cached expr |
| `compute(root, tip)` | `HTM` (numeric) | — wraps `compute_np` |
| `compute_np(root, tip)` | `NpMatrix4x4` | `@memoize` within one tick |
| `recompute()` | — | Clears cache, calls `evaluated_all_fks.evaluate()` |

## Related

- **Uses:** [[sdt.world_description.world_entity.Connection]], [[sdt.world_description.world_entity.KinematicStructureEntity]], [[sdt.world_description.world_state.WorldState]], [[concept.forward-kinematics]]
- **Used by:** [[sdt.world.World]] (delegated from `compute_forward_kinematics`)

## Open questions

- Whether `compose_expression` (symbolic) and `compute_np` (numeric) can diverge if `recompute()` has not been called since the last state change — e.g. if `compile()` is called after a model change but `recompute()` is not.

## Provenance

- `forward_kinematics.py:1-196` — `ForwardKinematicsManager` dataclass, BFS cache update, `compile`, `recompute`, `compose_expression`, `compute`, `compute_np`.
