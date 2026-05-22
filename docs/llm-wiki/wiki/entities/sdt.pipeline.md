---
id: sdt.pipeline
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/pipeline/pipeline.py
    lines: [1, 174]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/pipeline/mesh_decomposition/base.py
    lines: [1, 81]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.world_entity.Body
  - sdt.world_description.shape_collection.ShapeCollection
  - sdt.spatial_types.spatial_types
used_by:
  - sdt.adapters
status: stable
tags: [sdt, pipeline, step, mesh-decomposition, asset-processing, world-transform]
last_ingest: 2026-05-18
---

_A `Pipeline` is an ordered sequence of `Step` objects that transform a `World` in place; concrete steps include body filtering, mesh-centre normalisation, semantic annotation injection, and convex mesh decomposition._

## Purpose

The pipeline abstraction solves the asset-preparation problem: raw URDF/FBX/MJCF worlds often have geometry that is unsuitable for physics or perception (meshes not centred at the body origin, non-convex collision shapes, body names that don't match semantic annotation conventions). A `Pipeline` chains corrective `Step` objects so that the same preparation logic can be applied uniformly across dataset loaders and test fixtures.

## Architecture

```
Pipeline(steps=[step1, step2, step3])
    │
    └── world = step1.apply(world)
        world = step2.apply(world)
        world = step3.apply(world)
```

Each `Step.apply(world)` wraps `_apply(world)` in `world.modify_world()`, so every step is an atomic world modification.

## `Step` and `Pipeline`

| Class | Description |
|---|---|
| `Step` | Abstract `@dataclass`; subclasses implement `_apply(world) -> World`. |
| `Pipeline` | `@dataclass(steps: List[Step])`; `apply(world)` runs all steps in order. |

## Concrete steps

### `BodyFilter`

```python
BodyFilter(condition=lambda body: "link" in body.name.name)
```

Removes bodies that do not satisfy `condition`. Traverses `world.bodies` and calls `world.remove_kinematic_structure_entity(body)` for each rejected body.

### `CenterLocalGeometryAndPreserveWorldPose`

Adjusts each body's collision mesh vertices so the body origin is at the AABB centre of its collision geometry. Updates `parent_connection.parent_T_connection_expression` to compensate, preserving the body's world pose. Also adjusts all child connection transforms to maintain child world poses.

**When to use:** FBX parsers produce worlds where all bodies have an origin at (0, 0, 0) even though their collision meshes are offset. This step normalises the geometry so that body origins coincide with mesh centres — a requirement for correct FK-based collision checks.

### `BodyFactoryReplace`

```python
BodyFactoryReplace(
    annotation_creator=lambda body, world: MyAnnotation(body, world),
    body_condition=lambda body: re.compile(r"^dresser_\d+.*$").fullmatch(body.name.name),
)
```

Replaces matching bodies with semantic annotation sub-worlds. For each matching body:
1. `annotation_creator(body, world)` produces a `HasRootKinematicStructureEntity` annotation (typically a `SemanticAnnotation` with its own sub-bodies).
2. The annotation's body tree is moved to a new world via `world.move_branch_to_new_world()`.
3. The original body and its subtree are removed.
4. The new world is merged back via `world.merge_world(new_world, parent_connection)`.

The default `body_condition` matches names like `dresser_1`, `dresser_2a`, etc.

## Mesh decomposition (`mesh_decomposition/`)

`MeshDecomposer` is an abstract `Step` subclass that decomposes non-convex collision meshes into convex parts. Concrete implementations:

| Class | File | Backend |
|---|---|---|
| `BulletVHACD` | `bullet_vhacd.py` | PyBullet V-HACD |
| `VHACD` | `vhacd.py` | Standalone V-HACD |
| `CoACD` | `coacd.py` | CoACD approximate convex decomposition |
| `BoxDecomposer` | `box_decomposer.py` | Bounding-box approximation |

`MeshDecomposer._apply(world)` iterates `world.bodies`, calls `apply_to_body(body)` on each, which replaces `body.collision` with the decomposed `ShapeCollection`. Visual geometry is preserved unchanged.

`apply_to_mesh_and_save(mesh, output_path)` writes the decomposed result to an `.obj` file — useful for pre-baking decompositions as assets.

## `gltf_loader.py`

Separate module in `pipeline/` (not a `Step` subclass) providing GLTF/GLB file loading into SDT worlds. Not fully read; assumed to produce `World` objects similar to `URDFParser.parse()`.

## Related

**Uses:** [[sdt.world.World]], [[sdt.world_description.world_entity.Body]], [[sdt.world_description.shape_collection.ShapeCollection]], [[sdt.spatial_types.spatial_types]]

**Used by:** [[sdt.adapters]] (ProcTHOR pipelines use `Step` subclasses)

**See also:** [[sdt.world.World]] (`modify_world()` context manager), [[sdt.world_description.shape_collection.ShapeCollection]]

## Open questions

- `BodyFactoryReplace` has a hardcoded default `body_condition` matching `dresser_\d+` — this is clearly dataset-specific (PartNet-Mobility dresser category). Whether this default is intentional or an oversight left from a specific dataset ingest is unclear.
- `MeshDecomposer.apply_to_body` replaces `body.collision` with decomposed visual geometry — the comment says "visual" but the replacement target is `body.collision`. This may be intentional (decomposing the visual mesh to use as collision) or a copy-paste bug.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/pipeline/pipeline.py:1-174` at commit `0528d8cf3` — `Step`, `Pipeline`, `BodyFilter`, `CenterLocalGeometryAndPreserveWorldPose`, `BodyFactoryReplace`.
- `semantic_digital_twin/src/semantic_digital_twin/pipeline/mesh_decomposition/base.py:1-81` at commit `0528d8cf3` — `MeshDecomposer` abstract base + `apply_to_shape/body/_apply`.
