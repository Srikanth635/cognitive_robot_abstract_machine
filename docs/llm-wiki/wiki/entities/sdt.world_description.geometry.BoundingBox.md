---
id: sdt.world_description.geometry.BoundingBox
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/geometry.py
    lines: [1, 1242]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - pycram.perception.PerceptionQuery
  - sdt.reasoning.predicates
  - sdt.reasoning.robot_predicates
status: stable
tags: [sdt, geometry, bounding-box, spatial, frame-aware, random-events, interval, contains]
last_ingest: 2026-05-17
---

_Frame-aware axis-aligned bounding box: min/max extents anchored to a homogeneous transform `origin`; `contains()` always transforms the query point into the origin frame before checking._

## Purpose

`BoundingBox` is the spatial containment type used throughout SDT for region queries,
support detection, and placement validation. Unlike a plain min/max box, each `BoundingBox`
carries an `origin: HomogeneousTransformationMatrix` that ties the extents to a specific
reference frame. Containment queries transform the input point to that frame before checking,
so boxes defined in different frames compare correctly.

Intervals per axis are `random_events.SimpleInterval` objects shifted by the origin's
offset, enabling interval-algebra operations (`intersection_with`, overlap size tests) used
by `is_supported_by` in `sdt.reasoning.predicates`.

## Construction

```python
BoundingBox(
    min_x, min_y, min_z,   # local-frame lower bounds (float)
    max_x, max_y, max_z,   # local-frame upper bounds (float)
    origin: HomogeneousTransformationMatrix,  # frame anchor
)
```

Factory methods:
- `BoundingBox.from_mesh(mesh)` — computes AABB from trimesh vertex bounds.
- `BoundingBox.from_min_max(min_point, max_point, origin)` — convenience wrapper.

## Interval representation

```python
x_interval = SimpleInterval(min_x + origin.x, max_x + origin.x)
y_interval = SimpleInterval(min_y + origin.y, max_y + origin.y)
z_interval = SimpleInterval(min_z + origin.z, max_z + origin.z)
simple_event = x_interval × y_interval × z_interval  # 3D closed box
```

`origin.x/y/z` are the translation components of the origin frame.

## Key operations

### `contains(point: Point3) → bool`

1. Transforms `point` into `self.origin.reference_frame` via `origin.inverse()`.
2. Checks `simple_event.contains((x, y, z))` — closed (inclusive) intervals.

Used by `PerceptionQuery.from_world()` to keep only bodies whose global position falls
inside the query region.

### `transform_to_origin() → BoundingBox`

Projects all 8 corners to the current frame and returns a new AABB that tightly wraps the
projected corners. Used when reanchoring a box from one reference frame to another.

### `intersection_with(other: BoundingBox) → BoundingBox | None`

Computes axis-wise interval intersection. Returns `None` if any axis has an empty
intersection. Used by `is_supported_by` to check whether two bodies' bounding boxes
overlap in X and Y.

### `bloat(x, y, z) → BoundingBox`

Returns a new, larger `BoundingBox` (functional — does not modify `self`). Each axis
is expanded by ±half the given amount:
`BoundingBox(min_x - x/2, min_y - y/2, min_z - z/2, max_x + x/2, ...)`.

### `enlarge(x, y, z)` / `enlarge_all(d)`

In-place mutation equivalents of `bloat`. Prefer `bloat()` for functional style.

### `as_shape() → Box`

Converts to a `sdt.world_description.geometry.Box` shape with extents
`(max_x - min_x, max_y - min_y, max_z - min_z)`. Used for visualization or
collision-geometry creation.

## Hash and equality

The hash is computed from `(min_x, min_y, min_z, max_x, max_y, max_z)` only; the
`origin` frame is excluded. Two `BoundingBox` objects with the same extents but different
reference frames are hash-equal and compare equal — a potential gotcha if they are used
as dict keys or set members while semantically distinct.

## Related

**Uses:** _(none — only the `random_events` library and `HomogeneousTransformationMatrix` value type; no other wiki entities)_

**Used by:** [[pycram.perception.PerceptionQuery]], [[sdt.reasoning.predicates]], [[sdt.reasoning.robot_predicates]]

**See also:** [[sdt.world_description.world_entity.Region]] — full kinematic entity that wraps a `ShapeCollection`; its bounding box is a `BoundingBox` derived from the region's geometry

## Open questions

- Exact line numbers for the `BoundingBox` class within `geometry.py` (1242 lines, also
  containing `Color`, `Scale`, `Shape`, `Mesh`, `Sphere`, `Cylinder`, `Box`) were not
  captured during ingest. `source_paths` covers the full file.
- `origin` is excluded from hash — this can cause silent bugs if two boxes with the same
  extents but different frames are stored in the same set or used as dict keys.
- `transform_to_origin()` projects all 8 corners and fits a new AABB; for rotated frames
  this may produce a significantly larger box than the original (axis-alignment loss).

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/world_description/geometry.py:1-1242` —
  `BoundingBox` class and factory methods. File also contains `Color`, `Scale`, `Shape`,
  `Mesh`, `Sphere`, `Cylinder`, `Box`.
