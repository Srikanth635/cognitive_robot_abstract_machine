---
id: concept.world
kind: concept
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [296, 510]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [780, 880]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py
    lines: [1769, 1830]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - concept.forward-kinematics
  - concept.semantic-annotation
  - sdt.spatial_types.Pose
  - sdt.world.World
  - sdt.world_description.connections
  - sdt.world_description.world_entity.Body
  - sdt.world_description.world_entity.Connection
  - sdt.world_description.world_entity.KinematicStructureEntity
used_by: []
status: stable
tags: [concept, sdt, world, kinematic-structure, tree, connection, pose, symbolic]
last_ingest: 2026-05-17
---

_The SDT kinematic world: a tree of `Body`/`Region` nodes connected by typed `Connection` edges, where every pose is a frame-anchored CasADi 4×4 matrix enabling symbolic FK._

## What the world is

The SDT world is a directed graph (`rustworkx.PyDAG`) that must always be a **tree**:

```
n_nodes == n_edges + 1   AND   is_weakly_connected(graph)
```

Nodes are `KinematicStructureEntity` objects — either `Body` (a physical rigid body) or `Region`
(a spatial region with no mass). Edges are `Connection` objects — kinematic links between a parent
and child entity. The tree invariant is enforced by `World.validate()`.

## KinematicStructureEntity — the node type

`KinematicStructureEntity` is an abstract base. Its concrete subtypes are:

- **`Body`** — a "semantic atom": the smallest unit with semantic meaning. Carries `visual:
  ShapeCollection`, `collision: ShapeCollection`, and `inertial: Inertial`. Visual and collision
  shapes have poses relative to the body's own frame.
- **`Region`** — a spatial region (no mass/inertia). Used for workspace definitions, forbidden
  zones, etc.

Every entity tracks `index: int` — its node index in `World.kinematic_structure` — set by the
World on insertion.

## Connection — the three-transform kinematic link

A `Connection` encodes the relationship between parent and child with three transforms:

```
parent_frame
  └── parent_T_connection   (constant HomogeneousTransformationMatrix)
        └── _kinematics      (variable, e.g., revolute joint rotation, prismatic translation)
              └── connection_T_child  (constant HomogeneousTransformationMatrix)
                    └── child_frame
```

- `parent_T_connection_expression` and `connection_T_child_expression` must be **constant** at
  construction time (no CasADi free variables). The validator raises if they contain free variables.
- `_kinematics` is computed by the concrete `Connection` subclass (e.g., `RevoluteConnection`
  applies a joint-angle-parameterised rotation around the joint axis).
- Multiplying these three gives the full FK transform from parent to child:
  `parent_T_child = parent_T_connection @ _kinematics @ connection_T_child`.

Common concrete `Connection` subclasses: `FixedConnection`, `RevoluteConnection`,
`PrismaticConnection`, `Connection6DoF`. `FixedConnection` has identity `_kinematics`.
`Connection6DoF` (used by pycram's `PlaceAction`) gives the child 6 unconstrained DoFs — effectively
floating in world space.

## Pose — the frame-anchored symbolic transform

`Pose(SymbolicMathType, SpatialType)` is a 4×4 `HomogeneousTransformationMatrix` backed by a CasADi
SX expression internally. It combines a `Point3` (position) and `Quaternion` (orientation) and ties
the result to a `reference_frame: KinematicStructureEntity`.

Because `Pose` is a CasADi symbolic matrix, FK expressions (multiplying connection transforms from
root to a leaf body) stay as **symbolic expressions** until `.evaluate()` is called. This allows the
QP solver in giskardpy to differentiate through FK automatically.

Factory methods: `Pose.from_xyz_rpy(...)`, `Pose.from_xyz_quaternion(...)`.

## Semantic layer

On top of the geometric tree, SDT adds `semantic_annotations: Set[SemanticAnnotation]` to every
`WorldEntity`. These annotations are queryable via `krrood.entity_query_language` (EQL). The
`reasoning` subpackage evaluates EQL queries against the current world state. This is the mechanism
pycram uses for precondition/postcondition evaluation in `ActionDescription`.

## Thread-safe modification

All structural changes must go through the `World`'s modification protocol:

```python
with world.modify_world():
    world.move_branch_with_fixed_connection(body, new_parent_frame)
```

or via `@atomic_world_modification` on SDT-internal methods. See [[sdt.world.World]] for details.

## How pycram uses the world

1. `PickUpAction.execute()` calls `world.move_branch_with_fixed_connection(object, end_effector_frame)`
   — rewires the tree to make the grasped object a child of the end-effector body.
2. `PlaceAction.execute()` detaches the object by replacing its connection with a `Connection6DoF`
   to the world root — effectively "dropping" it at its current FK pose.
3. All motion designators access `world.root` to pick the FK base frame for their QP goals.

## Related

- World container: [[sdt.world.World]]
- Body node: [[sdt.world_description.world_entity.Body]]
- Kinematic connection: [[sdt.world_description.world_entity.Connection]]
- Node base type: [[sdt.world_description.world_entity.KinematicStructureEntity]]
- Pose type: [[sdt.spatial_types.Pose]]
- Package: [[sdt]]
- pycram coupling: [[bridge.pycram-sdt]]

## Open questions

- `Region` node type is referenced but not read in detail — its role beyond spatial labelling
  (e.g., forbidden zones, workspace regions) and whether it participates in FK is not confirmed.
  Phase 6 expansion target.
- CasADi-backed Pose enables symbolic FK differentiation, but the full FK traversal code lives in
  `spatial_computations/` — not yet ingested. Phase 6/7 target.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py:296-510` at
  commit `0528d8cf3` — `KinematicStructureEntity`, `Body` class definitions.
- Same file, lines 780–880 — `Connection` class, three-transform structure.
- `semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py:1769-1830` at
  same commit — `Pose` constructor, `reference_frame`, CasADi backing.
