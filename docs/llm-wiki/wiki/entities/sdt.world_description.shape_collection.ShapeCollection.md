---
id: sdt.world_description.shape_collection.ShapeCollection
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/shape_collection.py
    lines: [1, 150]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.world_entity.Body
  - sdt.world_description.world_entity.Region
  - sdt.semantic_annotations.SemanticAnnotations
  - sdt.adapters
  - sdt.pipeline
status: stable
tags: [sdt, shape, geometry, mesh, bounding-box, collision]
last_ingest: 2026-05-18
---

_Ordered list of `Shape` objects anchored to a `KinematicStructureEntity`; provides combined mesh, bounding-box queries, and frame transformations._

## Purpose

`ShapeCollection` is the geometry carrier for all kinematic entities. `Body` holds two — `visual` and `collision` — while `Region` holds one — `area`. The collection is the point of contact for both rendering and collision: `combined_mesh` merges all shapes for physics, and `as_bounding_box_collection_in_frame` drives perception bounding-box queries.

## When to use

- When adding geometry to a `Body` or `Region` — construct a `ShapeCollection([shape1, shape2, ...])`.
- When querying the bounding box of an entity in a given frame — call `as_bounding_box_collection_in_frame(frame).bounding_box`.
- When reading the merged collision mesh — use `combined_mesh`.

## Construction / dependencies

```python
sc = ShapeCollection([Box(...), Mesh(...)])
sc.reference_frame = body   # typically set by Body/Region __post_init__
sc.transform_all_shapes_to_own_frame()  # normalise shape origins

# Bounding-box query in robot root frame:
bbox = sc.as_bounding_box_collection_in_frame(robot.root).bounding_box
```

## Key attributes

| Attribute | Type | Notes |
|---|---|---|
| `shapes` | `List[Shape]` | Constituent shapes |
| `reference_frame` | `KinematicStructureEntity` | Anchor for frame transforms |
| `combined_mesh` | `trimesh.Trimesh` | `@cached_property`; merges all shape meshes |

## `BoundingBoxCollection`

A subclass of `ShapeCollection` whose shapes are all axis-aligned bounding boxes.

| Method | Returns | Notes |
|---|---|---|
| `bounding_box()` | single AABB | Encompasses all bounding-box shapes |
| `bloat(delta)` | `BoundingBoxCollection` | Expands each box by `delta` in all axes |
| `merge()` | `BoundingBoxCollection` | Collapses all boxes to one |
| `from_event(event)` | `BoundingBoxCollection` | Constructs from a perception event |

## Related

- **Used by:** [[sdt.world_description.world_entity.KinematicStructureEntity]] (abstract `combined_mesh`), [[sdt.world_description.world_entity.Body]] (`visual`/`collision`), [[sdt.world_description.world_entity.Region]] (`area`), [[sdt.semantic_annotations.SemanticAnnotations]] (factory creates `BoundingBoxCollection` shapes), [[sdt.adapters]] (URDF parser constructs `ShapeCollection` from geometry elements), [[sdt.pipeline]] (`MeshDecomposer` replaces `body.collision` with decomposed `ShapeCollection`)

## Provenance

- `shape_collection.py:1-150` — `ShapeCollection`, `BoundingBoxCollection`, `combined_mesh`, `as_bounding_box_collection_in_frame`, `bounding_box`, `bloat`, `merge`.
