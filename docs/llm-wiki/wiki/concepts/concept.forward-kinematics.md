---
id: concept.forward-kinematics
kind: concept
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_computations/forward_kinematics.py
    lines: [1, 196]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_state.py
    lines: [88, 350]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.Connection
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
  - sdt.world_description.world_state.WorldState
  - sdt.spatial_computations.forward_kinematics
used_by:
  - concept.world
  - concept.motion-statechart
  - sdt.spatial_computations.forward_kinematics
  - sdt.spatial_computations.ik_solver
status: stable
tags: [sdt, fk, kinematics, casadi, symbolic, batch, numpy, zero-copy]
last_ingest: 2026-05-17
---

_How SDT computes and evaluates transforms between any two nodes in the kinematic tree — symbolically for composition, numerically for the tick loop._

## Core idea

Every `Connection` exposes an `origin_expression`: a symbolic CasADi SX matrix whose free variables are exactly the DOF positions in `WorldState`. FK from root to any node is the product of all `origin_expression` terms along the path — assembled once on model change, compiled to a CasADi function, then evaluated in batch each tick by feeding `world.state.positions` as a numpy memory view.

This two-layer design (symbolic ↔ numeric) lets giskardpy differentiate the FK expressions analytically (for Jacobians) while the tick loop pays only an array-copy-free numpy evaluation.

## Symbolic layer

`ForwardKinematicsManager.update_root_T_kse_expression_cache()` walks the PyDAG via BFS and chains:

```python
root_T_child = root_T_parent @ connection.origin_expression
```

for every connection in the tree. The results are cached in `root_T_kse_expression_cache: Dict[UUID, HTM]`.

`compose_expression(root, tip)` handles arbitrary root→tip pairs by computing the split chain: invert the root-side edges, compose the tip-side edges. The result is `@copy_memoize`-cached — a memoized deep copy so callers get an independent symbolic expression.

## Numeric layer

`compile()` stacks all per-node FK expressions into a single `Matrix`, compiles it into a `CompiledFunction`, and binds the position row of `world.state._data` as a memory view:

```python
compiled_all_fks.bind_args_to_memory_view(0, world.state.positions)
```

`recompute()` then calls `compiled_all_fks.evaluate()` — the compiled function reads directly from the numpy array with no copy.

`compute_np(root, tip)` retrieves the pre-computed result from `forward_kinematics_for_all_bodies` (a `4N×4` array, 4 rows per body) using UUID-keyed row offsets. It is `@memoize`-cached within one tick; the cache is invalidated by `clear_memoization_cache` at the start of each `recompute()`.

## WorldState as the shared state bus

`WorldState._data` is the `4×N float64` array that backs all symbolic evaluation:
- Row 0: positions — passed to FK compiled functions
- Rows 1-3: velocities, accelerations, jerks — passed to giskardpy control loop

This single array is the ground truth for both SDT FK and giskardpy QP. Neither ever copies it; both bind to it at compile time.

## Invalidation contract

`ForwardKinematicsManager` is a `ModelChangeCallback`. When the world topology changes (nodes/connections added or removed), `_notify` re-runs `update_root_T_kse_expression_cache → compile → recompute`. State changes (joint position updates) do not recompile — only `recompute()` is called.

## FK chain for non-root pairs

If neither root nor tip is the world root, the computation is:

```
root_T_tip = inverse(map_T_root) @ map_T_tip
```

where both `map_T_*` slices come from `forward_kinematics_for_all_bodies`.

## Related

- **Uses:** [[sdt.world_description.world_entity.Connection]] (`origin_expression`), [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] (DOF variables), [[sdt.world_description.world_state.WorldState]] (state bus), [[sdt.spatial_computations.forward_kinematics]] (implementation)
- **Used by:** [[concept.world]], [[concept.motion-statechart]] (Jacobians via FK expressions), [[sdt.spatial_computations.ik_solver]]
