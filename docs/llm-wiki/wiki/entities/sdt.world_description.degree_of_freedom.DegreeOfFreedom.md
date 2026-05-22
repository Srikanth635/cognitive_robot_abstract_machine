---
id: sdt.world_description.degree_of_freedom.DegreeOfFreedom
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/degree_of_freedom.py
    lines: [1, 120]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - concept.forward-kinematics
  - sdt.world_description.world_entity.Connection
  - sdt.world_description.connections
  - bridge.sdt-giskardpy
  - giskardpy.qp.adapters
  - sdt.spatial_computations.ik_solver
  - sdt.world_description.world_state.WorldState
  - sdt.adapters
status: stable
tags: [sdt, dof, joint, state, variable, casadi]
last_ingest: 2026-05-18
---

_Scalar degree of freedom stored in the world state; carries symbolic CasADi variables for position, velocity, acceleration, and jerk._

## Purpose

Each joint coordinate in the SDT world is a `DegreeOfFreedom`. It owns a `DerivativeMap[FloatVariable]` whose entries are CasADi `SX` symbols that resolve their numerical values by reading `_world.state[dof.id]` at evaluation time. This lets FK expressions remain symbolic until evaluation while keeping all mutable state in one place (`World.state`).

## When to use

- When reading or writing a joint position: `dof.position = 1.2` writes to `world.state`.
- When building FK expressions that depend on the joint value: use `dof.variables.position` (a `FloatVariable` whose `resolve()` reads `world.state`).
- The `create_variables()` call is mandatory after the DOF is added to the world — before it, `variables` is empty.

## Construction / dependencies

```python
dof = DegreeOfFreedom(limits=DegreeOfFreedomLimits(...), has_hardware_interface=True)
world.add_dof(dof)          # sets _world and calls create_variables()
dof.position = 0.0          # shortcut: writes world.state[dof.id].position
```

## Key attributes

| Attribute | Type | Notes |
|---|---|---|
| `limits` | `DegreeOfFreedomLimits` | `lower/upper: DerivativeMap[float]` for pos/vel/acc/jerk |
| `variables` | `DerivativeMap[FloatVariable]` | Symbolic CasADi variables; valid only after `create_variables()` |
| `has_hardware_interface` | `bool` | Whether a real hardware actuator backs this DOF |

## `create_variables()`

Creates four `FloatVariable` entries in `variables` — `position`, `velocity`, `acceleration`, `jerk`. Each has a `resolve()` that reads `_world.state[dof.id].<derivative>`. Must be called once per DOF after the DOF has been assigned to a world.

## `DegreeOfFreedomLimits`

```python
@dataclass
class DegreeOfFreedomLimits:
    lower: DerivativeMap[float]   # lower bounds for pos/vel/acc/jerk
    upper: DerivativeMap[float]   # upper bounds for pos/vel/acc/jerk
```

`_overwrite_dof_limits(old, new)` returns the more restrictive of two limit sets (element-wise `max(lower)`, `min(upper)`).

## Related

- **Used by:** [[sdt.world_description.world_entity.Connection]], [[sdt.world_description.connections]], [[bridge.sdt-giskardpy]], [[giskardpy.qp.adapters]], [[sdt.spatial_computations.ik_solver]], [[sdt.world_description.world_state.WorldState]], [[sdt.adapters]]

## Provenance

- `degree_of_freedom.py:1-120` — `DegreeOfFreedom`, `DegreeOfFreedomLimits`, `create_variables()`, `_overwrite_dof_limits()`.
