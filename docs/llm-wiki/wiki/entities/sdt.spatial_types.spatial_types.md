---
id: sdt.spatial_types.spatial_types
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py
    lines: [1, 2078]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.spatial_types.Pose
used_by:
  - sdt.spatial_computations.forward_kinematics
  - sdt.spatial_computations.ik_solver
  - sdt.spatial_computations.raytracer
  - pycram.locations.costmaps
  - sdt.adapters
  - giskardpy.qp.adapters
  - sdt.pipeline
status: stable
tags: [sdt, spatial, htm, homogeneous-transform, casadi, symbolic, quaternion, rotation, point, vector]
last_ingest: 2026-05-18
---

_The foundational spatial algebra module for SDT: `HomogeneousTransformationMatrix` (HTM), `RotationMatrix`, `Point3`, `Vector3`, `Quaternion`, and the already-documented `Pose` — all backed by CasADi SX symbolic arithmetic and anchored to kinematic reference frames._

## Purpose

This module defines the numeric+symbolic spatial types that flow through every FK chain, QP constraint expression, and motion goal in the codebase. The CasADi SX backing means any arithmetic expression (matrix multiply, add, rotate) remains symbolic until `.evaluate()` is called — enabling giskardpy to differentiate FK chains for QP gradient construction without separate AD passes. All types carry an optional `reference_frame: KinematicStructureEntity` that tracks which kinematic node the value is expressed in; this lets FK composition (`world_T_tip = world_T_root @ root_T_link @ ...`) propagate frame annotations automatically.

## SpatialType base

`SpatialType` is the common `@dataclass` mixin. It holds:

- `casadi_sx: ca.SX` — the raw CasADi SX matrix.
- `reference_frame: Optional[KinematicStructureEntity]` — the coordinate frame.

`__deepcopy__` is overridden to copy the SX data but **keep the same `reference_frame` pointer** (frames are identity objects in the kinematic tree, not values).

`_ensure_consistent_frame(spatial_objects)` validates that all non-None inputs share one frame; raises `SpatialTypesError` on mismatch. Used internally by factory methods.

## `HomogeneousTransformationMatrix` (HTM)

4×4 matrix encoding translation + rotation. Inherits `SymbolicMathType`, `SpatialType`, `SubclassJSONSerializer`.

Additional attribute: `child_frame: Optional[KinematicStructureEntity]` — the frame whose origin this HTM locates (i.e. `reference_frame` T `child_frame`).

**Construction factory methods:**

| Method | Description |
|---|---|
| `from_point_rotation_matrix(point, rotation_matrix, ...)` | Compose from `Point3` + `RotationMatrix`. |
| `from_xyz_rpy(x, y, z, roll, pitch, yaw, ...)` | Euler angles (ZYX). |
| `from_xyz_quaternion(x, y, z, qx, qy, qz, qw, ...)` | Quaternion components. |
| `from_xyz_axis_angle(x, y, z, axis, angle, ...)` | Axis-angle rotation. |
| `create_with_variables(name, resolver)` | Symbolic HTM with named `FloatVariable` entries; used by FK binding layer. |

**Key operations:**

| Operation | Notes |
|---|---|
| `@` / `dot(other)` | Matrix multiply; preserves `reference_frame` from left, `child_frame` from right. |
| `inverse()` | Efficient HTM inverse: `R^T` in upper-left, `-R^T * t` in upper-right. |
| `to_position()` | Extract translation as `Point3`. |
| `to_rotation_matrix()` | Extract 3×3 rotation block as `RotationMatrix`. |
| `to_quaternion()` | `→ to_rotation_matrix().to_quaternion()`. |
| `to_pose()` | Cast to `Pose`. |
| `x`, `y`, `z` | Property accessors for the translation column. |

`__hash__` is defined for constant HTMs (no free variables): hashes `(position_xyz, quaternion_xyzw, reference_frame)`.

**CasADi note:** When data contains `FloatVariable` instances (symbolic DOF values), the HTM is a symbolic expression. `ca.mtimes` in `dot()` and `@` keeps the expression tree unevaluated until the CasADi function is compiled and called. This is the mechanism enabling differentiation in giskardpy.

## `RotationMatrix`

4×4 matrix with the upper-left 3×3 block populated and the fourth row/column fixed to `[0, 0, 0, 1]`. `_verify_type` enforces this at construction.

**Factory methods:**

| Method | Description |
|---|---|
| `from_axis_angle(axis, angle)` | Rodrigues formula via CasADi cos/sin. |
| `from_quaternion(q)` | Unit quaternion → rotation matrix (Orocos KDL formula). |
| `from_rpy(roll, pitch, yaw)` | ZYX Euler angles. |
| `from_vectors(x, y, z)` | Two of three orthogonal basis vectors; third is computed via cross product. |

**Key operations:** `@` / `dot()`, `inverse()` (= `.T`), `to_rpy()`, `to_quaternion()`, `to_axis_angle()`, `rotational_error(other)` (trace-based angular distance).

`x_vector()`, `y_vector()`, `z_vector()` extract the three basis-vector columns as `Vector3`.

## `Point3`

4-vector `[x, y, z, 1]` (homogeneous coordinate, last element = 1). Arithmetic:

- `Point3 + Vector3 → Point3`
- `Point3 - Point3 → Vector3`
- `Point3 - Vector3 → Point3`

Geometric helpers: `project_to_plane`, `project_to_line`, `distance_to_line_segment`, `euclidean_distance`, `norm`.

`to_vector3()` drops the homogeneous 1 → 0 (type conversion, no data copy).

## `Vector3`

4-vector `[x, y, z, 0]` (homogeneous coordinate, last element = 0). Has an extra `visualisation_frame` field used only by debug visualisation code.

Static basis constructors: `Vector3.X()`, `Vector3.Y()`, `Vector3.Z()`, `Vector3.NEGATIVE_X()`, etc.

Arithmetic: `+`, `-`, scalar `*` / `/`, `dot()` (`@`), `cross()`, `norm()`, `scale(a)` (normalise then multiply). `safe_division` avoids NaN by substituting a unit vector when the denominator is zero.

Advanced: `project_to_cone(axis, theta)` — symbolic cone projection with collinearity handling. `slerp(other, t)` — spherical linear interpolation of direction vectors.

## `Quaternion`

4-vector `[x, y, z, w]` (note: `w` at index 3, not 0).

**Factory methods:** `from_axis_angle`, `from_rpy`, `from_rotation_matrix`.

**Operations:** `conjugate()`, `multiply(q)` (Hamilton product), `diff(q)` (returns `p` such that `self * p = q`), `slerp(other, t)` (handles `q == -q` antipodal case), `to_axis_angle()`, `to_rotation_matrix()`, `to_rpy()`, `normalize()`.

**CasADi symbolic note:** `from_rpy` and `from_axis_angle` use `ca.cos`/`ca.sin` directly, producing CasADi SX expressions when the angles contain DOF variables. This is how revolute/prismatic joint transforms stay symbolic.

## `Pose`

Documented in full at [[sdt.spatial_types.Pose]]. It is a `HomogeneousTransformationMatrix` subclass with a simpler `__init__` (takes `Point3` + `Quaternion`). Factory methods `from_xyz_rpy` and `from_xyz_quaternion` delegate to `Point3` + `Quaternion` constructors.

## `rotation_matrix_to_quaternion`

Module-level function decorated with `@sm.substitution_cache`. Implements numerically stable matrix-to-quaternion conversion via `if_greater_zero` branches (avoids the degenerate trace == 0 case). The cache prevents CasADi subgraph duplication when this conversion is called inside FK expression construction.

## Related

**Uses:** [[sdt.world_description.world_entity.KinematicStructureEntity]], [[sdt.spatial_types.Pose]]

**Used by:** [[sdt.spatial_computations.forward_kinematics]], [[sdt.spatial_computations.ik_solver]], [[sdt.spatial_computations.raytracer]], [[pycram.locations.costmaps]], [[sdt.adapters]], [[giskardpy.qp.adapters]]

**See also:** [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]] (provides the symbolic scalar DOF values that make HTMs symbolic), [[concept.forward-kinematics]]

## Open questions

- `RotationMatrix.normalize()` calls `self[:3, col].scale(scale_v)` with `scale_v = 1.0` — normalising to unit length 1. This is a no-op for already-normalised matrices. Whether it is intended to detect drift and correct it (e.g. after many symbolic compositions) or is simply a placeholder is unclear.
- `Vector3.create_with_variables` and `Point3.create_with_variables` produce symbolic objects with named `FloatVariable` entries but the `resolver` closures use `lambda: resolver()[0]` etc. — capturing `resolver` by reference. If `resolver` is reassigned after `create_with_variables` returns, all three closures would resolve to the new object. This closure capture pattern is a latent bug if callers reuse the `resolver` variable.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py:1-2078` at commit `0528d8cf3` — `SpatialType`, `HomogeneousTransformationMatrix`, `RotationMatrix`, `Point3`, `Vector3`, `Quaternion`, `Pose`, `rotation_matrix_to_quaternion`, type aliases.
