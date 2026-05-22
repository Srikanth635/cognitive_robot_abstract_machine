---
id: sdt.world_description.connections
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/connections.py
    lines: [1, 1148]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.Connection
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
used_by:
  - concept.world
  - bridge.pycram-sdt
  - pycram.locations.locations.AccessingLocation
  - giskardpy.model.world_config
  - sdt.datastructures.joint_state.JointState
  - sdt.semantic_annotations.SemanticAnnotations
  - pycram.robot_plans.actions.core.container
  - sdt.adapters
status: stable
tags: [sdt, connection, joint, prismatic, revolute, 6dof, drive, kinematic]
last_ingest: 2026-05-18
---

_All concrete `Connection` subclasses: fixed, 1-DOF actives (prismatic/revolute), 6-DOF floating, and drive models (omni/differential)._

## Purpose

`connections.py` is the factory floor for SDT joints. Each class here sets `_kinematics` in its `add_to_world` or as a property getter, creating the variable FK term that lets the world evaluate transforms from symbolic DOF state. Drive models additionally implement `HasUpdateState` to reset odometry velocity DOFs each cycle.

## Hierarchy

```
Connection (base)
├── FixedConnection              — 0 DOF, identity _kinematics
├── ActiveConnection             — adds has_hardware_interface
│   ├── ActiveConnection1DOF     — single axis, multiplier/offset
│   │   ├── PrismaticConnection  — translation along axis
│   │   └── RevoluteConnection   — rotation about axis
│   ├── OmniDrive                — 7 DOF holonomic drive + HasUpdateState
│   └── DifferentialDrive        — 6 DOF non-holonomic drive + HasUpdateState
└── Connection6DoF               — 7 DOF free-floating joint
```

## FixedConnection

```python
FixedConnection(parent, child, parent_T_connection_expression=HTM())
```

`_kinematics` stays at identity (default `HomogeneousTransformationMatrix()`). The `create_with_dofs` factory returns `(self, [])` — no DOFs needed. Used by `PickUpAction` to attach a grasped object to the robot arm.

## ActiveConnection1DOF

Abstract; adds:

| Field | Type | Notes |
|---|---|---|
| `axis` | `Vector3` | Joint axis in connection frame |
| `multiplier` | `float` | Scales raw DOF value (default `1.0`) |
| `offset` | `float` | Added to scaled value (default `0.0`) |
| `dof_id` | `UUID` | References `world.state[dof_id]` |

`dof` property returns a deep-copy of the raw `DegreeOfFreedom` with limits rescaled by multiplier/offset. Position/velocity/acceleration/jerk R/W properties delegate to `_world.state[dof_id]`.

### PrismaticConnection

`add_to_world` sets `_kinematics = HTM.from_xyz_rpy(axis * dof.variables.position)`.
Translates child along `axis` by the DOF value.

### RevoluteConnection

`add_to_world` sets `_kinematics = HTM.from_xyz_axis_angle(axis, dof.variables.position)`.
Rotates child about `axis` by the DOF value.

## Connection6DoF

Seven UUID fields (`x, y, z, qx, qy, qz, qw`); models a completely unconstrained floating joint (e.g. a graspable object before attachment). `origin.setter` writes all seven DOF values at once. `create_with_dofs` allocates 7 `DegreeOfFreedom` objects and initialises `qw.position = 1.0` (identity quaternion). Used by `PlaceAction` to detach objects from the robot.

## OmniDrive

7 DOFs: `x, y, theta` (odometry position) + `vx, vy, vtheta` (velocity commands) + `steer` (optional). `_kinematics` = `odom_T_bf @ bf_T_bf_vel @ bf_vel_T_bf`. `update_state` resets all velocity position DOFs to `0` each control cycle so velocity commands are single-step.

## DifferentialDrive

6 DOFs: `x, y, theta` (odometry) + `v, omega` (velocity) + `wheel_angle` (passive). Same kinematic chain pattern as OmniDrive. `update_state` resets `v` and `omega` to `0`.

## Related

- **Uses:** [[sdt.world_description.world_entity.Connection]], [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]]
- **Used by:** [[concept.world]], [[bridge.pycram-sdt]], [[pycram.locations.locations.AccessingLocation]], [[giskardpy.model.world_config]], [[sdt.datastructures.joint_state.JointState]], [[sdt.semantic_annotations.SemanticAnnotations]], [[pycram.robot_plans.actions.core.container]], [[sdt.adapters]]

## Open questions

- `Connection6DoF` exposes both `dofs` and individual UUID properties. Whether `world.state` is the ground truth for both (and the UUID properties are just named shortcuts) needs verification on write paths.

## Provenance

- `connections.py:1-1148` — full hierarchy of concrete `Connection` subclasses.
