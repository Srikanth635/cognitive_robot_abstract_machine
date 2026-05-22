---
id: sdt.world_description.world_entity.Region
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [504, 545]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.shape_collection.ShapeCollection
used_by:
  - pycram.robot_plans.actions.core.misc
  - sdt.semantic_annotations.SemanticAnnotations
status: stable
tags: [sdt, world-entity, region, spatial, bounding-box, perception]
last_ingest: 2026-05-18
---

_Virtual `KinematicStructureEntity` representing a named spatial volume; carries an `area` `ShapeCollection` used for perception bounding-box queries._

## Purpose

A `Region` labels a volume of space in the kinematic tree — it has no physical mass or collision geometry. Its primary runtime use is to restrict object detection to a spatial sub-region: `DetectAction` calls `region.area.as_bounding_box_collection_in_frame(robot.root).bounding_box` to obtain a bounding box for the perception query.

Contrast with [[sdt.world_description.world_entity.Body]], which represents a rigid physical object.

## When to use

- Use when you want to semantically name a sub-volume of the world (e.g. "the shelf area", "the kitchen counter").
- Use as the `location` of a `DetectAction` or `SearchAction` to spatially gate perception.
- Do **not** use for objects with mass or collision geometry — use `Body` instead.

## Construction / dependencies

```python
# From a shape collection:
region = Region.from_shape_collection(PrefixedName("shelf_area"), shape_coll)

# From a point cloud (convex hull):
region = Region.from_3d_points(PrefixedName("counter_area"), points_3d)

world.add_node(region)
```

`__post_init__` sets `area.reference_frame = self` and normalises all shape origins to the region's local frame.

## Key attributes

| Attribute | Type | Notes |
|---|---|---|
| `area` | `ShapeCollection` | Shapes defining the spatial volume; `hash=False` |
| `combined_mesh` | `Optional[trimesh.Trimesh]` | Delegates to `area.combined_mesh` |

## Related

- **Uses:** [[sdt.world_description.world_entity.KinematicStructureEntity]], [[sdt.world_description.shape_collection.ShapeCollection]]
- **Used by:** [[pycram.robot_plans.actions.core.misc]] (bounding-box query in `DetectAction`), [[sdt.semantic_annotations.SemanticAnnotations]] (`HasRootRegion` mixin)

## Provenance

- `world_entity.py:504-545` — class definition, `area` field, `__post_init__`, `from_shape_collection`, `combined_mesh`, serialisation.
