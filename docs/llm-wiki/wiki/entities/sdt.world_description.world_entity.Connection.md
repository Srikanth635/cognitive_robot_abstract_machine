---
id: sdt.world_description.world_entity.Connection
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [779, 938]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
used_by:
  - concept.forward-kinematics
  - concept.world
  - sdt.world_description.connections
  - sdt.spatial_computations.forward_kinematics
  - sdt.adapters
status: stable
tags: [sdt, connection, kinematic, joint, three-transform, FK]
last_ingest: 2026-05-18
---

_Kinematic edge between a parent and child `KinematicStructureEntity` in the SDT world tree; encodes three sequential transforms: fixed parent frame → variable kinematics → fixed child frame._

## Purpose

`Connection` is the SDT equivalent of a URDF joint. It lives as a `PyDAG` edge in `World.kinematic_structure` and participates in every FK chain between two nodes. The three-transform split (`parent_T_connection @ _kinematics @ connection_T_child`) lets subclasses vary only the middle term while keeping the constant frame offsets immutable and separately serialisable.

## When to use

- Use as a type annotation when accepting any joint.
- Read `active_dofs` / `passive_dofs` to discover the degrees of freedom a joint controls.
- Use `origin_expression` for a symbolic (CasADi) form; use `origin` for the numerically evaluated FK result.

## Construction

```python
# Always constructed via a concrete subclass factory.
FixedConnection(parent=body_a, child=body_b, parent_T_connection_expression=htm)
Connection6DoF.create_with_dofs(parent=body_a, child=body_b, world=world)
```

`__post_init__` validates that both constant frames contain no free CasADi variables, then sets `reference_frame` / `child_frame` annotations on them.

## Key attributes

| Field | Type | Notes |
|---|---|---|
| `parent` | `KinematicStructureEntity` | Parent node |
| `child` | `KinematicStructureEntity` | Child node |
| `parent_T_connection_expression` | `HTM` | Constant; parent→connection fixed offset |
| `_kinematics` | `HTM` | Variable; set by subclass `add_to_world` or property |
| `connection_T_child_expression` | `HTM` | Constant; connection→child fixed offset |

## Key properties

| Property | Returns | Notes |
|---|---|---|
| `origin_expression` | symbolic `HTM` | `parent_T_connection @ _kinematics @ connection_T_child` |
| `origin` | evaluated `HTM` | `_world.compute_forward_kinematics(parent, child)` |
| `active_dofs` | `List[DegreeOfFreedom]` | `[]` in base; overridden by active connections |
| `passive_dofs` | `List[DegreeOfFreedom]` | `[]` in base; overridden by drive connections |
| `is_controlled` | `bool` | `False` in base |
| `has_hardware_interface` | `bool` | `False` in base |

## Subclasses

See [[sdt.world_description.connections]] for the full hierarchy:
- `FixedConnection` — 0 DOF, constant kinematics (identity)
- `ActiveConnection1DOF` → `PrismaticConnection`, `RevoluteConnection`
- `Connection6DoF` — 7 DOFs for free-floating bodies
- `OmniDrive`, `DifferentialDrive` — drive models with state integration

## Related

- **Uses:** [[sdt.world_description.world_entity.KinematicStructureEntity]], [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]]
- **Used by:** [[sdt.world_description.connections]], [[concept.world]], [[sdt.spatial_computations.forward_kinematics]]

## Provenance

- `world_entity.py:779-938` — class definition, fields, `__post_init__`, `origin_expression`, `origin`, `dofs`, `add_to_world`, default property implementations.
