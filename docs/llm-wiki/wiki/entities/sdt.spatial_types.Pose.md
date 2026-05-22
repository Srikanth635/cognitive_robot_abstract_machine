---
id: sdt.spatial_types.Pose
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py
    lines: [1769, 1850]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
used_by:
  - pycram.plans.failures.PlanFailure
  - bridge.pycram-sdt
  - bridge.sdt-giskardpy
  - concept.world
  - pycram.datastructures.ExecutionData
  - pycram.datastructures.grasp.GraspDescription
  - pycram.robot_plans.actions.core.NavigateAction
  - pycram.robot_plans.actions.core.PlaceAction
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.navigation
  - pycram.robot_plans.motions.robot_body
  - pycram.robot_plans.actions.composite
  - giskardpy.qp.adapters
  - sdt.spatial_types.spatial_types
status: stable
tags: [sdt, pose, spatial, homogeneous-transform, casadi, symbolic, reference-frame]
last_ingest: 2026-05-18
---

_A frame-anchored 4×4 homogeneous transformation matrix backed by a CasADi SX symbolic expression; the primary pose type throughout SDT and pycram motion designators._

## Purpose

`Pose` represents a 3D position + orientation tied to a specific `reference_frame` entity in the
kinematic tree. Internally it is a 4×4 `HomogeneousTransformationMatrix` using CasADi SX, which
means FK chains composed by matrix multiplication stay as symbolic expressions until `.evaluate()`
is called. This is what allows giskardpy to differentiate FK expressions for QP constraint construction.

In pycram, `Pose` appears as:
- The `goal_pose` parameter of `MoveToolCenterPointMotion` and other motion designators.
- The `current_pose` / `goal_pose` fields of `NavigationGoalNotReachedError` (in `failures.py`).
- The target of `NavigateAction.post_condition` (checked via `np.allclose(robot.root.global_pose, target)`).

## Construction

```python
Pose(
    position=Point3(x=1.0, y=0.0, z=0.5),
    orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
    reference_frame=some_body,
)
# or factory methods:
Pose.from_xyz_rpy(x, y, z, roll, pitch, yaw, reference_frame=frame)
Pose.from_xyz_quaternion(x, y, z, qx, qy, qz, qw, reference_frame=frame)
```

The `reference_frame` links the pose to a `KinematicStructureEntity`, so FK computations know
which frame this transform is expressed in.

## Key attributes

| Name | Notes |
|---|---|
| `reference_frame` | `Optional[KinematicStructureEntity]` — frame this pose is expressed in. `None` = world-absolute. |
| `_casadi_sx` | 4×4 CasADi SX matrix (from `HomogeneousTransformationMatrix`). Symbolic when containing free variables. |

## Related

- Reference frame type: [[sdt.world_description.world_entity.KinematicStructureEntity]]
- Concept: [[concept.world]]
- Uses in failure types: [[pycram.plans.failures.PlanFailure]]
- Uses in bridges: [[bridge.pycram-sdt]]

## Open questions

- `Pose.to_json()` raises `SpatialTypeNotJsonSerializable` if the pose contains free CasADi
  variables. It is unclear when symbolic (non-constant) poses arise at the pycram interface vs
  only internally in FK chains.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py:1769-1850` at
  commit `0528d8cf3` — `Pose.__init__`, `_verify_type`, `to_json`, `from_xyz_rpy`.
