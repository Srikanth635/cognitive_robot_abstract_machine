---
id: sdt.world_description.world_entity.Body
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [427, 502]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.shape_collection.ShapeCollection
used_by:
  - pycram.plans.failures.PlanFailure
  - bridge.pycram-sdt
  - concept.world
  - sdt.reasoning.predicates
  - sdt.reasoning.robot_predicates
  - pycram.querying.predicates
  - giskardpy.model.world_config
  - pycram.datastructures.grasp.GraspDescription
  - sdt.collision_checking
  - sdt.semantic_annotations.SemanticAnnotations
  - sdt.spatial_computations.raytracer
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.container
  - pycram.robot_plans.actions.core.container
  - pycram.robot_plans.actions.core.PickUpAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.actions.composite
  - pycram.locations.costmaps
  - sdt.adapters
  - sdt.pipeline
status: stable
tags: [sdt, body, entity, kinematic, semantic-atom, collision, visual, inertial]
last_ingest: 2026-05-18
---

_The SDT "semantic atom": an indivisible physical rigid body in the world tree, carrying visual/collision shapes and inertial properties._

## Purpose

`Body` is a `KinematicStructureEntity` that represents a rigid body — a link in robot or object
nomenclature. It is the leaf-level semantic unit; further decomposition would lose semantic meaning.
`Body` nodes are inserted into `World.kinematic_structure` (a `rustworkx.PyDAG`) and connected to
other bodies via `Connection` edges.

In pycram, the grasped object in `PickUpAction` and `PlaceAction` is a `Body`. It is re-parented
in the kinematic tree on grasp (attached to the end-effector) and on release (detached to world root
with a `Connection6DoF`).

## Key attributes

| Name | Kind | Notes |
|---|---|---|
| `visual` | `ShapeCollection` | Shapes for rendering. Poses are relative to the body's own frame. Set as `reference_frame = self` in `__post_init__`. |
| `collision` | `ShapeCollection` | Shapes for collision checking. Same frame convention as `visual`. |
| `inertial` | `Optional[Inertial]` | Mass and inertia tensor. Used by physics simulation adapters. |
| `index` | `Optional[int]` | PyDAG node index in `World.kinematic_structure`; set by the World on insertion. |

Inherited from `KinematicStructureEntity`: `name: PrefixedName`, `id: UUID`, `_world`, `_semantic_annotations`.

## Key methods

| Name | Notes |
|---|---|
| `combined_mesh` | Property. Merges all collision shapes into a single trimesh. Returns `None` if no collision geometry. |
| `has_collision(...)` | True if collision geometry exists and exceeds volume/surface thresholds. |
| `get_semantic_annotations_by_type(type_)` | Filters `_semantic_annotations` by type — used for semantic queries. |
| `from_shape_collection(name, shape_collection)` | Factory: creates a Body where visual == collision == the given collection. |

## Related

- Node base: [[sdt.world_description.world_entity.KinematicStructureEntity]]
- Concept: [[concept.world]]
- pycram coupling: [[bridge.pycram-sdt]]
- Failure type using Body: [[pycram.plans.failures.PlanFailure]]

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py:427-502` at
  commit `0528d8cf3` — full `Body` class definition.
