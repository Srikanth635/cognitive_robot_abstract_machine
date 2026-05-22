---
id: sdt.world_description.world_entity.KinematicStructureEntity
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [296, 424]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.shape_collection.ShapeCollection
  - sdt.world.World
used_by:
  - bridge.sdt-giskardpy
  - concept.semantic-annotation
  - concept.world
  - sdt.world_description.world_entity.Body
  - sdt.world_description.world_entity.Region
  - sdt.world_description.world_entity.Connection
  - giskardpy.model.world_config
  - sdt.reasoning.WorldReasoner
  - sdt.semantic_annotations.SemanticAnnotations
  - sdt.spatial_computations.forward_kinematics
  - sdt.spatial_computations.ik_solver
  - sdt.spatial_types.Pose
  - sdt.spatial_types.spatial_types
status: stable
tags: [sdt, kinematic, entity, abstract, node, rustworkx]
last_ingest: 2026-05-18
---

_Abstract base class for all nodes in the SDT kinematic tree; the shared contract between `Body` (rigid objects) and `Region` (virtual volumes)._

## Purpose

Every node in `World.kinematic_structure` (`rustworkx.PyDAG`) is a `KinematicStructureEntity`. The class provides the FK interface (global pose, transform, CoM), the world backref, and the graph index so that the World can route FK queries without knowing the concrete subtype.

## When to use

- When a function accepts any kinematic node — use `KinematicStructureEntity` as the type hint.
- When traversing the tree (children, parent, ancestor connections) — all traversal methods live here.
- Do **not** instantiate directly — always construct a `Body` or `Region`.

## Construction / dependencies

```python
# Never instantiated directly.
# Body(name=..., collision=..., visual=...) is the standard concrete constructor.
# _world and index are set by World.add_node — do NOT pass them at construction.
```

## Key attributes

| Attribute | Type | Notes |
|---|---|---|
| `_world` | `Optional[World]` | Back-reference; set by World, not __init__ |
| `index` | `Optional[int]` | PyDAG node index; set on insertion, cleared on removal |
| `combined_mesh` | `Optional[trimesh.Trimesh]` | Abstract; Body returns collision mesh, Region returns area mesh |

## Key properties

| Property | Description |
|---|---|
| `global_transform` | `_world.compute_forward_kinematics(world.root, self)` |
| `global_pose` | Same, converted to `Pose` |
| `center_of_mass` | `combined_mesh.center_mass` transformed to world frame |
| `parent_connection` | `Connection` edge pointing to this node's parent |
| `parent_kinematic_structure_entity` | Parent node in the DAG |
| `child_kinematic_structure_entities` | Direct children list |

`get_first_parent_connection_of_type(T)` walks ancestors until it finds a `Connection` of type `T`; raises `ValueError` at the root.

## Subclasses

- [[sdt.world_description.world_entity.Body]] — rigid object with `visual`/`collision` ShapeCollections and inertia.
- [[sdt.world_description.world_entity.Region]] — virtual volume with `area` ShapeCollection; labels space rather than matter.

## Related

- **Uses:** [[sdt.world_description.shape_collection.ShapeCollection]] (abstract `combined_mesh` contract), [[sdt.world.World]] (backref, FK delegation)
- **Used by:** [[sdt.world_description.world_entity.Connection]] (parent/child), [[concept.world]]

## Open questions

- `from_3d_points` factory belongs here (not on `Region` specifically) even though it is primarily used for regions — it calls `from_shape_collection` which is abstract, so any `KinematicStructureEntity` subclass can use it.

## Provenance

- `world_entity.py:296-424` — class definition, fields, properties, traversal methods.
