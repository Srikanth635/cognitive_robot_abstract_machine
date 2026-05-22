---
id: sdt.adapters
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/adapters/urdf.py
    lines: [1, 366]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/adapters/mjcf.py
    lines: [1, 60]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.world_entity.Body
  - sdt.world_description.world_entity.Connection
  - sdt.world_description.connections
  - sdt.world_description.degree_of_freedom.DegreeOfFreedom
  - sdt.world_description.shape_collection.ShapeCollection
  - sdt.spatial_types.spatial_types
  - sdt.pipeline
used_by:
  - giskardpy.model.world_config
  - sdt.robots.concrete
status: stable
tags: [sdt, adapter, urdf, mjcf, usd, fbx, loader, world-builder]
last_ingest: 2026-05-18
---

_The `adapters/` subpackage provides format-specific parsers that construct `sdt.world.World` objects from external robot/scene description files: URDF (primary), MJCF (MuJoCo), USD, FBX, and dataset loaders (PartNet-Mobility, SAGE-10K, ProcTHOR)._

## Purpose

The adapters are the entry point for importing robots and scenes into SDT. Rather than constructing `Body`, `Connection`, and `DegreeOfFreedom` objects by hand, callers parse a file via the appropriate adapter to get a fully wired `World`. The resulting `World` carries the kinematic tree, DOF variables, visual/collision geometry, and joint limits; `AbstractRobot.from_world()` is then called on top to wrap it with semantic annotations.

## Format coverage

| File/subpackage | Format | Notes |
|---|---|---|
| `urdf.py` | URDF / Xacro | Primary format; most complete; used by all concrete robot models. |
| `mjcf.py` | MuJoCo MJCF | Full MuJoCo XML including actuators, tendons, equality constraints, geoms. |
| `usd.py` | USD (Pixar) | Universal Scene Description; breadth unknown from file header. |
| `fbx.py` | FBX | Autodesk FBX; likely used for imported object meshes. |
| `mesh.py` | Mesh utilities | Helpers for loading trimesh-compatible mesh files. |
| `multi_sim.py` | Multi-simulator types | Shared dataclasses for MuJoCo actuators, geoms, cameras, joints used by `mjcf.py`. |
| `package_resolver.py` | ROS/filesystem URIs | `PathResolver` + `CompositePathResolver` for resolving `package://` and absolute paths. |
| `world_entity_kwargs_tracker.py` | JSON deserialisation | Tracks KSE ID-to-object mapping when reconstructing world from JSON. |
| `partnet_mobility_dataset/` | PartNet-Mobility | Dataset-specific loader + auto-generated semantic annotations for furniture. |
| `sage_10k_dataset/` | SAGE-10K | Dataset schema + loader for SAGE 10K object collection. |
| `procthor/` | ProcTHOR | Procedural indoor scene parser + pipeline integration. |
| `ros/` | ROS 2 | TF publisher, world synchroniser, message converters (SDT ↔ ROS 2). |

## `URDFParser` (primary adapter)

```python
world = URDFParser.from_file("path/to/robot.urdf").parse()
# or from xacro:
world = URDFParser.from_xacro("path/to/robot.urdf.xacro").parse()
# or from string:
world = URDFParser(urdf=urdf_string, prefix="my_robot").parse()
```

`parse()` workflow:

1. All `<link>` elements → `Body` objects via `parse_link()` (visual + collision `ShapeCollection`).
2. Root link identified from `urdfpy` tree.
3. `world.add_kinematic_structure_entity(root)` inside `world.modify_world()`.
4. All `<joint>` elements → `Connection` subclasses via `parse_joint()`, then `world.add_connection()`.

`parse_joint()` maps URDF joint types:

| URDF type | Connection subclass |
|---|---|
| `revolute`, `continuous` | `RevoluteConnection` |
| `prismatic` | `PrismaticConnection` |
| `fixed` | `FixedConnection` |
| (others) | Base `Connection` |

Joint limits (position + velocity), safety-controller overrides, and `<mimic>` joint relations are handled by `urdf_joint_to_limits()`. A mimic joint reuses an existing `DegreeOfFreedom` (looked up by the mimicked joint name) and adds a `multiplier`/`offset` to the connection. `NegativeConnectionVelocity` is raised if velocity limit < 0.

`parse_geometry()` maps URDF primitive types to SDT geometry: `Box` → `Box`, `Sphere` → `Sphere`, `Cylinder` → `Cylinder`, `Mesh` → `Mesh` (with URI resolved via `PathResolver`). Each geometry gets an `origin_transform` (relative to the link body) from the URDF `<origin>` element.

`prefix` is auto-detected from the URDF `<robot name>` if not provided; all entity names are `PrefixedName(name, prefix)` to avoid collisions in multi-robot worlds.

`CompositePathResolver` is the default URI resolver; it supports `package://` ROS URIs and falls back to filesystem paths.

## `package_resolver.py`

`PathResolver` is an abstract base; `CompositePathResolver` chains multiple resolvers in priority order. Responsible for translating `package://robot_name/path/to/mesh.obj` → absolute filesystem path. Used by URDF and MJCF parsers.

## Dataset adapters

- `partnet_mobility_dataset/loader.py` — loads PartNet-Mobility objects into SDT worlds; `generated_semantic_annotations.py` contains auto-generated annotation classes for each object category.
- `sage_10k_dataset/loader.py` + `schema.py` — SAGE dataset format with a JSON schema for validation.
- `procthor/procthor_parser.py` + `procthor_pipelines.py` — parser converts ProcTHOR JSON to SDT world; pipelines wire the `Pipeline`/`Step` machinery from `sdt.pipeline`.

## ROS adapter (`ros/`)

Provides bidirectional conversion between SDT and ROS 2:

- `msg_converter.py` / `ros2_to_semdt_converters.py` / `semdt_to_ros2_converters.py` — message type converters.
- `tf_publisher.py` + `tfwrapper.py` — publishes SDT kinematic tree as TF transforms.
- `world_synchroniser.py` + `world_fetcher.py` — live world state synchronisation with a ROS 2 node.
- `visualization/viz_marker.py` — publishes visual markers for RViz.

## Related

**Uses:** [[sdt.world.World]], [[sdt.world_description.world_entity.Body]], [[sdt.world_description.world_entity.Connection]], [[sdt.world_description.connections]], [[sdt.world_description.degree_of_freedom.DegreeOfFreedom]], [[sdt.world_description.shape_collection.ShapeCollection]], [[sdt.spatial_types.spatial_types]], [[sdt.pipeline]]

**Used by:** [[giskardpy.model.world_config]], [[sdt.robots.concrete]]

**See also:** [[sdt.world.World]], [[sdt.robots.abstract_robot.AbstractRobot]]

## Open questions

- `mjcf.py`, `usd.py`, and `fbx.py` were not read in full; their parse patterns may differ from `urdf.py`. In particular, MJCF supports `<actuator>` and `<tendon>` elements that have no URDF equivalent — whether these map to new `Connection` subclasses or are stored separately is unconfirmed.
- `world_entity_kwargs_tracker.py` (`WorldEntityWithIDKwargsTracker`) is used in `spatial_types.py` for JSON deserialisation of spatial types that carry `reference_frame_id`. The full ID-resolution protocol (how IDs map back to live KSE objects) is not traced here.

## Provenance

- `semantic_digital_twin/src/semantic_digital_twin/adapters/urdf.py:1-366` at commit `0528d8cf3` — `URDFParser`, `urdf_joint_to_limits`, `connection_type_map`.
- `semantic_digital_twin/src/semantic_digital_twin/adapters/mjcf.py:1-60` at commit `0528d8cf3` — imports and top-level structure (headers only).
