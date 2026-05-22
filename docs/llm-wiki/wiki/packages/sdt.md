---
id: sdt
kind: package
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/__init__.py
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by: []
status: stable
tags: [sdt, package, world, kinematic-structure, robot, semantic]
last_ingest: 2026-05-17
---

_Semantic Digital Twin (v0.0.6): a symbolic kinematic world model for robot simulation and reasoning, with a thread-safe tree-structured world graph, semantic annotations, and FK/IK._

## Purpose

SDT provides a **world model** that pycram and giskardpy depend on at runtime. Its core abstraction
is a directed acyclic kinematic structure (a `rustworkx.PyDAG`) whose nodes are `KinematicStructureEntity`
objects (bodies, regions) and whose edges are `Connection` objects (joints, fixed links). The tree must
always satisfy: `n_nodes == n_edges + 1` and weak connectivity.

SDT layers semantic annotations and reasoning on top of this geometric core, enabling symbolic
queries like "which object is in the robot's hand?" as well as FK/IK, collision checking, and adapter
import/export for common robot description formats.

## Subpackage tour

| Subpackage | Role |
|---|---|
| `world` (`world.py`) | [[sdt.world.World]] — the runtime world container; kinematic graph + atomic modification |
| `world_description` | [[sdt.world_description.world_entity.Body]], [[sdt.world_description.world_entity.Connection]], `Region`, `DegreeOfFreedom`, `ShapeCollection`; the entity and connection dataclasses |
| `spatial_types` | [[sdt.spatial_types.Pose]], `Point3`, `Quaternion`, `HomogeneousTransformationMatrix` — all implemented as CasADi symbolic matrices |
| `robots` | [[sdt.robots.abstract_robot.AbstractRobot]] + 17 pre-configured robot models (Panda, UR5, Tiago, HSR-B, Armar, …) |
| `spatial_computations` | Forward kinematics, inverse kinematics solver, raytracer |
| `semantic_annotations` | Semantic annotation classes layered over world entities; queried via krrood EQL |
| `reasoning` | Reasoning engine that evaluates EQL queries against the world |
| `datastructures` | `JointState`, `PrefixedName`, krrood variable wrappers |
| `collision_checking` | Collision detection (wraps geometry + trimesh) |
| `adapters` | Import/export: URDF, MJCF, USD, FBX, mesh loaders, package path resolver |
| `callbacks` | Event/callback system for world state changes |
| `orm` | Object-relational mapping layer |
| `pipeline` | Data pipeline utilities |

## Key entry points for cross-package consumers

- **pycram** accesses SDT through `Context.world: World` and `Context.robot: AbstractRobot`. See [[bridge.pycram-sdt]].
- **giskardpy** receives `Task` objects from pycram motion designators; these reference SDT poses internally. See [[bridge.pycram-giskardpy]].

## Core concept

The kinematic structure is a tree of `Body` / `Region` nodes connected by typed `Connection` edges.
Each `Connection` encodes three transforms: the fixed parent frame, variable kinematics (e.g., joint
rotation), and the fixed child frame. `Pose` is a CasADi 4×4 matrix anchored to a reference
`KinematicStructureEntity`, enabling symbolic FK expressions. See [[concept.world]].

## Related

- World model: [[sdt.world.World]]
- Robot model: [[sdt.robots.abstract_robot.AbstractRobot]]
- Core concept: [[concept.world]]
- Cross-package interface: [[bridge.pycram-sdt]]

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/__init__.py` at commit `0528d8cf3` — version 0.0.6.
- `world.py`, `robots/abstract_robot.py`, `world_description/world_entity.py`, `spatial_types/spatial_types.py` at same commit — source of all entity pages.
