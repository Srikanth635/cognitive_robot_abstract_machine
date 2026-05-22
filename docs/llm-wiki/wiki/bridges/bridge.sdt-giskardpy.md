---
id: bridge.sdt-giskardpy
kind: bridge
package: cross
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/context.py
    lines: [1, 119]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/binding_policy.py
    lines: [1, 81]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/qp/adapters/qp_adapter.py
    lines: [1, 50]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/qp/constraint_collection.py
    lines: [20, 30]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
  - sdt.spatial_types.Pose
  - sdt.collision_checking
  - giskardpy.motion_statechart.context
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - giskardpy.qp.qp_controller
  - giskardpy.qp.adapters
used_by: []
status: stable
tags: [bridge, sdt, giskardpy, world, dof, fk, collision, qp, context]
last_ingest: 2026-05-17
---

_giskardpy depends deeply on SDT throughout its QP and MSC layers: SDT's `World` is the execution substrate, `DegreeOfFreedom` objects are the QP optimization variables, and SDT spatial types are the constraint language._

## Purpose

giskardpy is not self-contained. It delegates world representation, kinematics, and
collision infrastructure entirely to SDT. This bridge is **one-directional**: giskardpy
imports SDT at every layer; SDT does not import giskardpy.

This contrasts with [[bridge.pycram-giskardpy]], which is the pycram→giskardpy
coupling (motion designators → Tasks → MSC execution). Here the coupling is
giskardpy→SDT.

## Coupling inventory

### 1. Execution context — `MotionStatechartContext` carries `sdt.world.World`

`giskardpy.motion_statechart.context.MotionStatechartContext` holds an `sdt.world.World`
reference as its primary field. Everything that touches world state (FK lookup, collision,
control command application) does so through this world object.

```python
context = MotionStatechartContext(
    world=sdt_world,
    qp_controller_config=QPControllerConfig(target_frequency=50),
)
executor.compile(msc)  # world FK symbols flow into QP from here
executor.tick()        # world.apply_control_commands() writes back
```

### 2. DOF objects are the QP optimization variables

`QPDataSymbolic.from_giskard(degrees_of_freedom, ...)` takes `List[sdt.world_description.degree_of_freedom.DegreeOfFreedom]` as its first argument. Each `DegreeOfFreedom` provides:
- `variables.velocity / acceleration / jerk` — the CasADi symbolic variables optimized
- `limits.upper.velocity / acceleration / jerk` — used to compute MPC velocity profiles
- `has_position_limits()` — determines whether position constraint rows are added

The QP's size ∝ number of active DOFs, not world complexity (via `_set_active_dofs`
which intersects DOF variables with constraint free variables).

### 3. FK binding — SDT kinematics frozen into float variables

`ForwardKinematicsBinding` (in `binding_policy.py`) creates a 3×4 auxiliary matrix of
`FloatVariable` entries backed by `FloatVariableData`. On each `bind(world)` call, it
calls `world.compute_forward_kinematics_np(root, tip)` and writes the numerical result
into the float variables. This enables tasks to refer to "current TCP pose" without
recomputing FK inside the CasADi expression — the expression references the float variables,
and FK is evaluated separately each tick.

Root and tip are `sdt.world_description.world_entity.KinematicStructureEntity` objects.

### 4. Spatial types as constraint language

`qp/constraint_collection.py` factory methods (`add_point_goal_constraints`,
`add_rotation_goal_constraints`) accept `sdt.spatial_types.Point3 / Vector3 / RotationMatrix`
directly as symbolic expressions. Goal constructors in
`giskardpy.motion_statechart.goals.*` take `sdt.spatial_types.Pose` and
`sdt.world_description.world_entity.Body / Connection` objects as goal targets.

### 5. Collision avoidance — giskardpy registers against SDT collision infrastructure

`MotionStatechartContext.self_collision_manager` and `external_collision_manager` are
`sdt.collision_checking.CollisionVariableManager` instances registered into
`sdt.collision_checking.CollisionManager` (obtained via `world.collision_manager`).
Collision avoidance goals (`goals/collision_avoidance.py`) read `ClosestPoints` data
from SDT's collision infrastructure to build repulsion constraints.

## Key observations

- giskardpy has no internal world model — all kinematic and geometric state is delegated
  to SDT. `DegreeOfFreedom` objects are the only joint state abstraction used; there is no
  giskardpy-internal joint config structure.
- `WorldStateTrajectoryPlotter` in `executor.py` (imported from `sdt`) is used for
  optional debug trajectory visualization — even the debug tooling is SDT.
- The `-0.0001` rad rotation singularity hack in `ConstraintCollection.add_rotation_goal_constraints`
  is identical to a constant in SDT's IK solver — this coupling is informal (same numeric
  constant, independently present) rather than a shared function.

## Related

- **Context:** [[giskardpy.motion_statechart.context]]
- **QP adapter:** [[giskardpy.qp.adapters]]
- **World:** [[sdt.world.World]]
- **DOF:** [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]]
- **KSE:** [[sdt.world_description.world_entity.KinematicStructureEntity]]
- **Collision:** [[sdt.collision_checking]]
- **Symmetric bridge (pycram↔giskardpy):** [[bridge.pycram-giskardpy]]

## Open questions

- Whether the `DegreeOfFreedom` list passed to `QPDataSymbolic` comes from the same SDT
  `WorldState` object that the executor writes to needs verification — if they are decoupled,
  jerk/acc limits could diverge at runtime.

## Resolved

- **WorldConfig vs pycram world (resolved 2026-05-17):** `giskardpy/model/world_config.py` is
  giskardpy's **standalone initialization path** — used only in the ROS 2 middleware / tests /
  scripts. When giskardpy is driven via pycram, the world is created externally and injected into
  `pycram.datastructures.Context.world`. It then flows:
  `Context.world → plan.world → ActionNode.construct_motion_state_chart() → MotionExecutor(world=plan.world) → MotionStatechartContext(world=self.world)`.
  A single `sdt.world.World` object is shared throughout — no dual-world issue.
  See [[giskardpy.model.world_config]].

## Provenance

- `giskardpy/src/giskardpy/motion_statechart/context.py:1-119` — `MotionStatechartContext`; collision manager lazy init.
- `giskardpy/src/giskardpy/motion_statechart/binding_policy.py:1-81` — `ForwardKinematicsBinding`, `GoalBindingPolicy`.
- `giskardpy/src/giskardpy/qp/adapters/qp_adapter.py:1-50` — imports of `DegreeOfFreedom`, `Derivatives`, `DerivativeMap`.
- `giskardpy/src/giskardpy/qp/constraint_collection.py:20-30` — SDT spatial type imports.
