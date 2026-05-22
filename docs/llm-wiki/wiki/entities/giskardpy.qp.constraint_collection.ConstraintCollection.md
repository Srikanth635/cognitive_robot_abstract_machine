---
id: giskardpy.qp.constraint_collection.ConstraintCollection
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/qp/constraint_collection.py
    lines: [1, 520]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - concept.qp-controller
  - giskardpy.qp.constraint
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
used_by:
  - giskardpy.motion_statechart.graph_node.Task
  - giskardpy.qp.qp_controller
  - concept.qp-controller
  - giskardpy.qp.adapters
status: stable
tags: [giskardpy, qp, constraint-collection, factory, lifecycle-gating]
last_ingest: 2026-05-18
---

_Container for all QP constraints produced by a `Task.build()` call; provides high-level factory methods for common motion constraint patterns._

## Purpose

`ConstraintCollection` accumulates symbolic QP constraints during `build()`. After `build()`, the collection is linked to its owner node (lifecycle gating), then merged into the global collection. All constraint names must be unique within a collection.

## Lifecycle gating

```python
def link_to_motion_statechart_node(self, node: MotionStatechartNode):
    is_running = if_eq(node.life_cycle_variable, RUNNING, 1, 0)
    for c in self._constraints:
        c.quadratic_weight *= is_running
```

Called automatically from `MotionStatechart._build_and_apply_artifacts()`. After this call, all constraint weights become zero when the node is not RUNNING — inactive constraints cost nothing in the QP without matrix rebuilding.

## Merging

```python
combined = ConstraintCollection()
for node in msc.nodes:
    combined.merge(name_prefix=node.unique_name, other=node._constraint_collection)
```

`merge(prefix, other)` prepends `"prefix/"` to every constraint name in `other`, then appends them to `self`. Raises `DuplicateNameException` if any name is already present after prefixing.

## Factory methods

All factory methods call `add_constraint(BaseConstraint)` or `add_equality_constraint`/`add_inequality_constraint` internally.

### Position-level (integral) constraints

| Method | What it adds |
|---|---|
| `add_equality_constraint(task_expression, equality_bound, reference_velocity, quadratic_weight)` | `EqualityConstraint`: drives `expression → expression + bound` |
| `add_inequality_constraint(task_expression, lower_error, upper_error, reference_velocity, quadratic_weight)` | `InequalityConstraint`: keeps expression in `[lower, upper]` range |
| `add_position_constraint(expr_current, expr_goal, reference_velocity)` | shorthand equality with `bound = expr_goal - expr_current` |
| `add_position_range_constraint(expr_current, expr_min, expr_max, reference_velocity)` | shorthand inequality |
| `add_point_goal_constraints(frame_P_current, frame_P_goal, reference_velocity, weight)` | adds 3 equality constraints, one per axis of a 3D point |
| `add_vector_goal_constraints(frame_V_current, frame_V_goal, reference_velocity)` | aligns two unit vectors; uses SLERP intermediate to avoid singularity at π |
| `add_rotation_goal_constraints(frame_R_current, frame_R_goal, reference_velocity)` | aligns rotation matrices; uses quaternion representation; adds 3 constraints (xyz, w redundant) |

### Derivative (velocity-level) constraints

| Method | What it adds |
|---|---|
| `add_velocity_constraint(lower, upper, task_expression, velocity_limit)` | `DerivativeInequalityConstraint`: keeps derivative of expression in `[lower, upper]` |
| `add_velocity_eq_constraint(velocity_goal, task_expression, velocity_limit)` | `DerivativeEqualityConstraint`: targets specific velocity |
| `add_velocity_eq_constraint_vector(…)` | batch version of `add_velocity_eq_constraint` |
| `add_translational_velocity_limit(frame_P_current, max_velocity)` | limits translation speed via norm constraint |
| `add_rotational_velocity_limit(frame_R_current, max_velocity)` | limits rotation speed via axis-angle magnitude constraint |

## Rotation singularity hack

`add_rotation_goal_constraints` applies a `-0.0001 rad` rotation around Z before computing quaternion error:
```python
hack = RotationMatrix.from_axis_angle(Vector3.Z(), -0.0001)
frame_R_current = frame_R_current.dot(hack)
```
This nudges the rotation away from the quaternion singularity at zero rotation. The same constant `-0.0001` appears in `sdt.spatial_computations.ik_solver` for the same reason.

## Related

- **Uses:** [[giskardpy.qp.constraint]], [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- **Used by:** [[giskardpy.motion_statechart.graph_node.Task]] (produces `NodeArtifacts.constraints`), [[giskardpy.qp.qp_controller]], [[concept.qp-controller]]

## Provenance

- `constraint_collection.py:1-520` — full class including all factory methods.
