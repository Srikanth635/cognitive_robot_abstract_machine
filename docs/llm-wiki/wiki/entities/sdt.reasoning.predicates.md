---
id: sdt.reasoning.predicates
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/predicates.py
    lines: [1, 615]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.spatial_computations.raytracer
  - sdt.collision_checking
  - sdt.world_description.world_entity.Body
  - sdt.world_description.geometry.BoundingBox
  - sdt.robots.abstract_robot.AbstractRobot
used_by:
  - pycram.perception.PerceptionQuery
  - sdt.reasoning.WorldReasoner
  - sdt.reasoning.robot_predicates
status: stable
tags: [sdt, reasoning, predicate, symbolic-function, visible, contact, stable, supported-by, spatial-relation, eql]
last_ingest: 2026-05-17
---

_Module of `@symbolic_function`-decorated predicates for spatial, contact, and semantic reasoning over the SDT world; usable both as concrete Python functions and lazy EQL expressions._

## Purpose

`sdt.reasoning.predicates` is the low-level inference layer consumed by `PerceptionQuery`,
the `WorldReasoner`, and EQL query expressions. Every function is decorated with
`@symbolic_function`, meaning it returns concrete results when called with `Body`/concrete
arguments and returns a lazy `InstantiatedVariable` when called with EQL `Variable` arguments.

The module covers: visibility (`visible`, `get_visible_bodies`, `occluding_bodies`), contact
(`contact`), support (`is_supported_by`, `is_supporting`), reachability (`reachable`),
placement (`is_place_occupied`), region containment (`is_body_in_region`, `InsideOf`), type
testing (`ContainsType`), and six world-frame spatial relations.

## Core predicates

### `visible(camera: Camera, obj: Body) → bool`

Returns `True` if `obj` appears in `get_visible_bodies(camera)`.

**Critical — orientation is ignored.** `get_visible_bodies` builds a camera pose using
only position:

```python
cam_pose[:3, 3] = camera.root.global_transform.to_np()[:3, 3]
# rotation block left as np.eye(3) — camera facing direction discarded
```

`RayTracer.create_segmentation_mask(cam_pose, resolution=256)` then sees a camera at the
correct world position but always facing the default identity direction. All bodies that
appear in the 256-pixel segmentation from that vantage are reported as visible regardless
of the camera's actual facing direction.

Used by `PerceptionQuery.from_world()` as its final visibility filter.

### `stable(obj: Body) → bool`

Raises `NotImplementedError("Needs multiverse")` unconditionally. **Dead branch** — any
code path reaching this function will fail at runtime until a physics multiverse backend
is integrated.

### `contact(body1: Body, body2: Body, threshold: float = 0.001) → bool`

Calls `FCLCollisionDetector.check_collision_between_bodies(body1, body2)`. Returns `True`
if the reported distance ≤ `threshold`. Returns `False` if the collision manager returns
`None` (bodies not tracked). Threshold default of 1 mm captures near-contact states.

### `reachable(pose: Pose, root: KSE, tip: KSE) → bool`

Calls `world.compute_inverse_kinematics(root, tip, pose)`. Catches `MaxIterationsException`
and `UnreachableException`; returns `False` on either, `True` on convergence. The IK
computation is delegated to `sdt.spatial_computations.ik_solver` through the World API —
there is no direct `InverseKinematicsSolver` import here.

### `is_supported_by(supported: Body, supporting: HasSupportingSurface, max_h: float = 0.1) → bool`

1. `Below()(supporting.root_body, supported)` — verifies the support surface is geometrically
   below the supported object (z-ordering check).
2. Computes `BoundingBox` intersection per axis via `random_events.SimpleInterval` arithmetic.
3. Checks x and y intervals overlap **and** z-interval overlap size < `max_h` (objects sharing
   a thin z-boundary, not stacked or fully overlapping).

`is_supporting(surface, obj)` is the symmetric inverse — delegates to `is_supported_by(obj, surface)`.

### `occluding_bodies(camera: Camera, body: Body) → List[Body]`

Creates an isolated world copy containing only `body`, renders two segmentation masks
(full world vs. body-only), and returns all bodies appearing in the full-world mask that
are absent or reduced in the isolated mask. Identifies what is occluding `body`.

### `compute_euclidean_planar_distance(b1: Body, b2: Body, dim: int = 2) → float`

Returns XY-plane Euclidean distance between two bodies' CoM positions.

## Placement and region predicates

### `is_place_occupied(box: BoundingBox, world: World, allowed_bodies: List[Body]) → bool`

Builds a trimesh `CollisionManager`, adds all world bodies transformed to world frame
(excluding `allowed_bodies`), and checks the `box` mesh against the manager. Returns `True`
on first collision found. Useful for checking whether a target placement volume is free.

### `is_body_in_region(body: Body, region: Region) → float`

Trimesh boolean intersection between `body.collision_mesh` and `region.area.mesh`.
Returns fractional volume of `body` inside `region` (0.0 to 1.0).

### `InsideOf(a: Body, b: Body) → float`

Returns the fraction of `a`'s mesh vertices that lie inside `b`'s bounding box. Not a
`@symbolic_function`; used as a geometric helper.

### `ContainsType(Predicate)`

EQL `Predicate` subclass. `ContainsType(iterable, obj_type)` returns `True` if
`any(isinstance(obj, obj_type) for obj in iterable)`.

## Spatial relation classes

All are callable frozen dataclasses using `_signed_distance_along_direction(axis_index)`.
Positive signed distance = the named relation holds.

| Class | Axis | Direction |
|---|---|---|
| `LeftOf(a, b)` | X=0 | `a` is left of `b` in world frame |
| `RightOf(a, b)` | X=0 | `a` is right of `b` |
| `Above(a, b)` | Z=2 | `a` is above `b` |
| `Below(a, b)` | Z=2 | `a` is below `b` |
| `Behind(a, b)` | Y=1 | `a` is behind `b` |
| `InFrontOf(a, b)` | Y=1 | `a` is in front of `b` |

Three base class variants: `KinematicStructureEntitySpatialRelation` (body-to-body),
`PointSpatialRelation` (point-to-body), `ViewDependentSpatialRelation` (camera-relative).

## Design observations

- `visible()` ignores camera orientation — it detects bodies reachable from the camera
  position in any direction, not specifically what the camera faces. This differs from
  `pycram.pose_validator.visibility_validator()` which uses a point-to-point `ray_test`.
- `stable()` is dead code at this commit; any plan depending on physics stability will
  fail at runtime.
- Spatial relations use fixed world-frame axes — `LeftOf`/`RightOf` do not account for
  the robot's heading unless `ViewDependentSpatialRelation` is used.

## Related

**Uses:** [[sdt.world.World]], [[sdt.spatial_computations.raytracer]], [[sdt.collision_checking]], [[sdt.world_description.world_entity.Body]], [[sdt.world_description.geometry.BoundingBox]], [[sdt.robots.abstract_robot.AbstractRobot]]

**Used by:** [[pycram.perception.PerceptionQuery]], [[sdt.reasoning.WorldReasoner]], [[sdt.reasoning.robot_predicates]]

**See also:** [[krrood.entity_query_language.predicate.Predicate]], [[pycram.pose_validator]]

## Open questions

- `visible()` uses an identity-rotation camera pose, so a camera facing away from an
  object still reports it as visible if it is within range of the segmentation viewport.
  Whether this is intentional ("object is detectable if the robot turns toward it") or a
  bug is undocumented.
- `stable()` raises unconditionally. The multiverse backend is mentioned in the error
  message but no issue or branch tracks its implementation.
- The `256` resolution for `create_segmentation_mask` is hardcoded inside
  `get_visible_bodies`. No per-call API to adjust it, making large or distant scenes
  potentially under-resolved.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/reasoning/predicates.py:1-615` — all
  predicates, spatial relation classes, `InsideOf`, `ContainsType`, `is_place_occupied`,
  `is_body_in_region`, `get_visible_bodies`, `occluding_bodies`.
