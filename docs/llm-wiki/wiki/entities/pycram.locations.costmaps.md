---
id: pycram.locations.costmaps
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/locations/costmaps.py
    lines: [1, 861]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.world_entity.Body
  - sdt.spatial_computations.raytracer
  - sdt.spatial_types.spatial_types
status: stable
tags: [location, costmap, occupancy, visibility, gaussian, ring, sampling, orientation]
last_ingest: 2026-05-18
used_by:
  - pycram.locations.locations.CostmapLocation
  - pycram.locations.locations.AccessingLocation
---

_Five costmap types — `Costmap` (base), `OccupancyCostmap`, `VisibilityCostmap`, `GaussianCostmap`, `RingCostmap` — plus `OrientationGenerator`; the spatial probability-distribution machinery underlying all pycram base-pose samplers._

## Purpose

Costmaps encode "how good is this 2D grid cell as a robot base position?" as a normalised numpy array. Each subclass computes its scores for a different geometric criterion. The base `Costmap.__add__` (aliasing `merge()`) ANDs two costmaps: a cell must be positive in **both** to appear in the output. `CostmapLocation` and `AccessingLocation` compose these primitives at construction time to produce the merged distribution they then sample from.

## Base class: `Costmap`

`Costmap` is a `@dataclass` with:

| Attribute | Type | Notes |
|---|---|---|
| `resolution` | `float` | Metres per grid cell. |
| `height`, `width` | `Optional[int]` | Grid dimensions in cells. |
| `origin` | `Pose` | World pose at the centre of the grid. |
| `map` | `np.ndarray` | 2D float array; positive = valid, zero = blocked. |
| `world` | `World` | World the costmap was built from. |
| `number_of_samples` | `int` | Max samples the iterator yields (default 200). |
| `sample_randomly` | `bool` | Random cell pick vs. top-N by score (default False). |
| `orientation_generator` | `Callable` | Overrides default facing-origin orientation. |

Key methods:

- `merge(other)` / `__add__(other)` — element-wise AND (both positive) → product; result normalised to [0, 1]. Raises `ValueError` if grids differ in size, origin, resolution, or world.
- `__iter__()` — calls `segment_map()`, splits sample budget across segments, picks top-N (or random) indices, converts to world `Pose` via `origin + (index − center) * resolution`. Delegates orientation to `orientation_generator`.
- `segment_map()` — connected-component labelling (`skimage.measure.label`, `connectivity=2`) so samples distribute across disconnected free regions.
- `partitioning_rectangles()` — axis-aligned bounding-rectangle decomposition; used internally by visualisation helpers.

## Subclasses

### `OccupancyCostmap`

Marks grid cells above which the world has no obstacles. Construction parameters (beyond base):

| Parameter | Notes |
|---|---|
| `distance_to_obstacle` | Clearance margin in metres; inflated via stride-trick sliding window. |
| `robot_view` | `AbstractRobot`; cells occupied by the robot itself are excluded from the blocked set. |

Algorithm: `create_ray_mask_around_origin()` casts vertical rays (z = `robot.base.bounding_box.height` → z = 0) over all grid cells using `RayTracer.ray_test`. Cells that hit something other than the robot are blocked. `inflate_obstacles()` erodes the free region by `distance_to_obstacle` using strided sub-matrix sums.

**When to use:** always — this is the foundation layer for every location sampler.

### `VisibilityCostmap`

Scores cells by how visible a target point is from that cell. Based on Mösenlechner PhD Thesis (TUM, page 173–178). Parameters beyond base:

| Parameter | Notes |
|---|---|
| `min_height` | Minimum camera height to consider (metres). |
| `max_height` | Maximum camera height to consider (metres). |
| `target_object` | `Body` or `Pose`; the point whose visibility is evaluated. |

Algorithm: creates four depth images (one per 90° sector) via `RayTracer.create_depth_map`, then for each cell computes which depth-image column and row range to consult, checks whether the observed depth exceeds the Euclidean distance to the cell, and sums the passing pixels. Score is normalised by the max. The four-image split and coordinate mapping are non-trivial; see source lines 570–753.

**Caveat (source comment):** The quaternion used for rotation between the four depth renders is technically invalid (never normalised). The existing tests pass with it, but any 90° yaw fails — the current implementation is acknowledged as an open issue in the source comment.

**When to use:** add to `OccupancyCostmap` when the robot must have line-of-sight to a target (e.g. `CostmapLocation(visible=True)`).

### `GaussianCostmap`

A 2D Gaussian centred on the origin with a hollow centre. Parameters beyond base:

| Parameter | Notes |
|---|---|
| `mean` | Side length of the square grid in cells; also the distribution mean. |
| `sigma` | Gaussian standard deviation. |

The map is `outer(gaussian_window, gaussian_window)`. The central 5% (`cut_dist = int(0.05 * mean)`) is zeroed out to discourage the robot from standing directly at the target.

**When to use:** `AccessingLocation` uses this to favour positions at a natural manipulation distance from a container without enforcing a strict ring.

### `RingCostmap`

A donut-shaped distribution — Gaussian in radial distance from the origin. Parameters beyond base:

| Parameter | Notes |
|---|---|
| `std` | Gaussian standard deviation in pixel units. |
| `distance` | Radius of the ring centre in metres. |

`ring()` computes Euclidean distance from each cell to the centre, then applies `exp(-((d - radius_px)² / (2 * std²)))`.

**When to use:** `CostmapLocation` (when `reachable=True`) uses `RingCostmap(distance=0.4)` — hardcoded 0.4 m radius — to concentrate candidates at a manipulation-friendly distance from the target.

## `OrientationGenerator`

Static utility class with three orientation strategies:

| Method | Description |
|---|---|
| `generate_origin_orientation(position, origin, rotate_by_angle)` | Robot faces `origin` from `position`. Default for most location samplers. |
| `orientation_generator_for_axis(axis)` | Returns a generator where the named axis faces the target instead of the robot forward direction. |
| `generate_random_orientation(rng)` | Random yaw from a seeded `random.Random(42)` — reproducible. |

## Combination pattern

```python
occ = OccupancyCostmap(resolution=0.1, distance_to_obstacle=0.2,
                       robot_view=robot, width=200, height=200,
                       origin=target_pose, world=world)
vis = VisibilityCostmap(min_height=0.8, max_height=2.0,
                        width=200, height=200, resolution=0.1,
                        origin=target_pose, world=world)
merged = occ + vis        # cells must pass both criteria
for pose in merged:       # iterate merged costmap for candidates
    ...
```

Merge fails with `ValueError` if the two costmaps have different size, origin, resolution, or world.

## Related

**Uses:** [[sdt.world.World]], [[sdt.world_description.world_entity.Body]], [[sdt.spatial_computations.raytracer]], [[sdt.spatial_types.spatial_types]]

**Used by:** [[pycram.locations.locations.CostmapLocation]], [[pycram.locations.locations.AccessingLocation]]

**See also:** [[pycram.locations.locations.GiskardLocation]], [[pycram.pose_validator]]

## Open questions

- The `VisibilityCostmap` quaternion bug (non-normalised rotation across the four depth renders) is acknowledged in a source comment but unfixed. The practical impact — incorrect depth-image orientation for 90° sectors — may cause the costmap to silently score cells incorrectly in those sectors.
- `OccupancyCostmap.create_ray_mask_around_origin` uses `self.width` as both the x and y grid size, making the grid always square. The `height` attribute from the base class is unused here.

## Provenance

- `pycram/src/pycram/locations/costmaps.py:1-861` at commit `0528d8cf3` — `OrientationGenerator`, `Rectangle`, `Costmap`, `OccupancyCostmap`, `VisibilityCostmap`, `GaussianCostmap`, `RingCostmap`, `plot_grid`.
