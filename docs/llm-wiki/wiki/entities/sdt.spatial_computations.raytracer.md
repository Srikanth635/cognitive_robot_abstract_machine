---
id: sdt.spatial_computations.raytracer
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_computations/raytracer.py
    lines: [1, 281]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.Body
  - sdt.world.World
used_by:
  - sdt.reasoning.predicates
  - sdt.reasoning.robot_predicates
  - pycram.locations.costmaps
status: stable
tags: [sdt, raytracer, perception, trimesh, depth-map, segmentation]
last_ingest: 2026-05-17
---

_`RayTracer`: a trimesh-backed ray casting engine that mirrors the SDT world as a scene graph and produces depth maps and segmentation masks from a camera pose._

## Purpose

The `RayTracer` maintains a `trimesh.Scene` in sync with `World.bodies` and their collision meshes. It exposes two rendering outputs from a camera pose: a depth map (float32 per pixel) and a segmentation mask (int32 per pixel, body index or -1). These are used for perception simulation — e.g. determining which bodies a robot's camera can see from a given viewpoint.

## Lazy update model

`RayTracer` tracks two version counters:
- `_last_world_model` — updated when `world.get_world_model_manager().version` changes (topology). Triggers `add_missing_bodies()`.
- `_last_world_state` — updated when `world.state.version` changes (joint positions). Triggers `update_transforms()`.

`update_scene()` is called at the top of every render method; it no-ops if both versions are current.

## Key methods

| Method | Returns | Notes |
|---|---|---|
| `create_segmentation_mask(camera_pose, resolution)` | `np.ndarray[int32]` | Body index or -1 per pixel |
| `create_depth_map(camera_pose, resolution)` | `np.ndarray[float32]` | Distance or -1 per pixel |
| `ray_test(origins, targets, multiple_hits)` | `(points, ray_indices, bodies)` | Raw trimesh intersection |
| `create_camera_rays(camera_pose, resolution, fov)` | `(origins, directions, pixels)` | Sets up perspective camera |

**Camera orientation convention:** The trimesh default camera looks along `-z`; `create_camera_rays` applies a `+90° Y-rotation` and `+180° X-rotation` to repoint it along `+x`.

## Scene management

`add_missing_bodies()` adds each collision geometry as a named node (`body.name_collision_i`) under `"world"`. The transform is `compute_forward_kinematics_np(root, body) @ collision.origin.to_np()`.

`update_transforms()` refreshes all collision node transforms when the state changes.

`scene_to_index` maps scene node names → world body indices; `index_to_body` maps body indices → `Body` objects. Together they translate trimesh triangle node names back to SDT bodies.

## Related

- **Uses:** [[sdt.world_description.world_entity.Body]], [[sdt.world.World]]
- **Used by:** [[sdt.reasoning.predicates]] (`get_visible_bodies()` calls `create_segmentation_mask()`), [[pycram.locations.costmaps]] (`OccupancyCostmap` and `VisibilityCostmap` use `RayTracer.ray_test` and `create_depth_map`)

## Open questions

- No `ModelChangeCallback` or `StateChangeCallback` registration is visible — `update_scene()` must be called manually before each render. Lazy invalidation via `version` counters avoids spurious recomputation but requires callers to not cache a stale `RayTracer` across world mutations.

## Provenance

- `raytracer.py:1-281` — `RayTracer` class, lazy update model, `add_missing_bodies`, `update_transforms`, `create_segmentation_mask`, `create_depth_map`, `create_camera_rays`, `ray_test`.
