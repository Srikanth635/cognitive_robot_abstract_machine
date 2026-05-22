---
id: sdt.world.World
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world.py
    lines: [1, 1961]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_state.WorldState
  - sdt.spatial_computations.forward_kinematics
  - sdt.spatial_computations.ik_solver
used_by:
  - pycram.datastructures.Context
  - bridge.pycram-sdt
  - concept.world
  - sdt.world_description.world_entity.KinematicStructureEntity
  - bridge.sdt-giskardpy
  - giskardpy.motion_statechart.context
  - giskardpy.model.world_config
  - pycram.locations.locations.CostmapLocation
  - pycram.locations.locations.AccessingLocation
  - pycram.locations.locations.GiskardLocation
  - pycram.perception.PerceptionQuery
  - pycram.pose_validator
  - sdt.reasoning.predicates
  - pycram.motion_executor.MotionExecutor
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.robots.concrete
  - sdt.collision_checking
  - sdt.reasoning.WorldReasoner
  - sdt.spatial_computations.raytracer
  - pycram.locations.costmaps
  - sdt.adapters
  - sdt.pipeline
status: stable
tags: [sdt, world, kinematic-structure, rustworkx, dag, atomic, thread-safe]
last_ingest: 2026-05-18
---

_The SDT world model: a rustworkx `PyDAG` of kinematic entities with thread-safe atomic modification and FK-preserving branch operations._

## Purpose

`World` is the central runtime state of `semantic_digital_twin`. It holds the full kinematic
structure of the robot's environment as a directed acyclic graph (`rx.PyDAG`). Every object,
link, joint, and region is a node; every connection (fixed, revolute, prismatic, 6-DoF, …) is
a directed edge. The world must be a **tree** at all times — a single root node, every other
node reachable from it, and exactly `n_nodes - 1` edges.

pycram accesses the world through `Context.world`. Modifications (attachment, detachment, pose
updates) must go through `world.modify_world()` or functions decorated with
`@atomic_world_modification` to acquire the internal RLock and maintain version numbering.

## Key attributes

| Name | Kind | Notes |
|---|---|---|
| `kinematic_structure` | `rx.PyDAG[KinematicStructureEntity]` | Core graph. All structural queries use rustworkx APIs on this field. |
| `_world_lock` | `threading.RLock` | Guards all structural modifications. Re-entrant so nested atomic calls within a single thread work safely. |
| `state` | `WorldState` | Snapshottable world state (joint positions, gripper states, etc.). |
| `_model_manager` | `WorldModelManager` | Tracks `version: int` and `model_modification_blocks: List[...]`. Incremented on every atomic modification. |

## Key methods

| Name | Notes |
|---|---|
| `modify_world()` | Returns a `WorldModelUpdateContextManager` that acquires `_world_lock`, clears memoization caches, and deletes orphaned DoFs on exit. Use as `with world.modify_world():`. |
| `validate()` | Asserts `len(nodes) == len(edges) + 1` and `rx.is_weakly_connected(kinematic_structure)`. Raises if either fails. |
| `root` | Memoized property. The unique node with in-degree 0. Raises if none or more than one exist. |
| `move_branch_with_fixed_connection(branch_root, new_parent)` | Computes FK for `branch_root`, removes its current parent edge, inserts a new `FixedConnection` to `new_parent`. Decorated `@atomic_world_modification`. Used by `PickUpAction` to attach a grasped object to the end-effector frame. |
| `__deepcopy__` | Custom. Creates a new `World` and copies bodies, regions, DoFs, and connections. Does NOT use `copy.deepcopy` on the graph — structured clone. |

## Modification protocol

Two equivalent patterns:

```python
# Pattern 1 — explicit context manager
with world.modify_world():
    world.move_branch_with_fixed_connection(object_body, end_effector_frame)

# Pattern 2 — decorator (used internally by SDT methods)
@atomic_world_modification
def my_operation(world, ...):
    ...
```

`@atomic_world_modification` raises if a nested atomic context is already active; nested calls
must use `modify_world()` directly.

## Related

- Context carrier: [[pycram.datastructures.Context]]
- SDT robot: [[sdt.robots.abstract_robot.AbstractRobot]]
- Bridge: [[bridge.pycram-sdt]]
- IK solver: [[sdt.spatial_computations.ik_solver]]
- Loaders: [[sdt.adapters]]
- Asset pipeline: [[sdt.pipeline]]
- Costmap machinery: [[pycram.locations.costmaps]]

## Open questions

- `World.__deepcopy__` performs a structured clone, but it is unclear if `semantic_annotations`
  and `actuators` fields are deep-copied or shallow-copied. Affects correctness of planning in
  sandbox/hypothetical worlds.
- SDT internals (`KinematicStructureEntity`, `WorldModelManager`, `HasSimulatorProperties`,
  `degrees_of_freedom`, `actuators`) not yet ingested — Phase 6 target.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/world.py:1-1961` at commit `0528d8cf3` —
  full module: `World`, `WorldModelUpdateContextManager`, `WorldModelManager`,
  `@atomic_world_modification`.
