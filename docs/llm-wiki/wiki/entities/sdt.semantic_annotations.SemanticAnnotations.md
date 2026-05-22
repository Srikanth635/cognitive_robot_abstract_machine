---
id: sdt.semantic_annotations.SemanticAnnotations
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/semantic_annotations.py
    lines: [1, 120]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/mixins.py
    lines: [1, 120]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.world_description.world_entity.Body
  - sdt.world_description.world_entity.Region
  - sdt.world_description.connections
  - sdt.world_description.shape_collection.ShapeCollection
used_by:
  - concept.semantic-annotation
  - pycram.robot_plans.actions.core.misc
status: stable
tags: [sdt, semantic, annotation, furniture, container, handle, door, drawer, perceivable]
last_ingest: 2026-05-17
---

_Bundled overview of all concrete `SemanticAnnotation` subclasses and their compositional mixin traits; covers `Furniture`, `Handle`, `Drawer`, `Door`, container types, and perceivable objects._

## Purpose

`semantic_annotations.py` provides the ready-to-use annotation library. Robot plans operate on annotation types — `PickUpAction` finds a `HasRootBody` annotation, `OpenAction` expects a `HasDoors` annotation. Annotations are the semantic vocabulary that bridges the geometric world to robot task knowledge.

## Mixin traits (from `mixins.py`)

| Mixin | Key fields | Semantics |
|---|---|---|
| `HasRootKinematicStructureEntity` | `root: KSE` | Base for HasRootBody / HasRootRegion |
| `HasRootBody` | `root: Body` | Primary rigid body; `scale`, `min_max_points` from mesh bounds |
| `HasRootRegion` | `root: Region` | Primary spatial volume |
| `IsPerceivable` | `class_label: str` | Perception class name for detection matching |
| `HasSupportingSurface` | `surface: Region`, `objects: List` | Table surface + inferred objects on top |
| `HasStorageSpace` | `storage_region: Region` | Interior volume for containment queries |
| `HasDrawers` | `drawers: List[Drawer]` | List of Drawer sub-annotations |
| `HasDoors` | `doors: List[Door]` | List of Door sub-annotations |
| `HasHandle` | `handle: Handle` | Graspable handle reference |
| `HasHinge` | `hinge: Hinge` | RevoluteConnection-based joint |
| `HasSlider` | `slider: Slider` | PrismaticConnection-based joint |
| `HasApertures` | `apertures: List[Aperture]` | Opening/closing apertures |
| `HasCaseAsRootBody` | `case: Body` | Outer casing body |

## Concrete annotation classes

| Class | Mixins | Typical use |
|---|---|---|
| `Furniture` | `SemanticAnnotation, ABC` | Base for all furniture; never instantiated directly |
| `Handle` | `HasRootBody` | Graspable handle geometry; `create_with_new_body_in_world` builds hollow box |
| `Drawer` | `HasRootBody, HasHandle, HasStorageSpace` | Sliding container with handle |
| `Door` | `HasRootBody, HasHandle, HasHinge` | Hinged opening with handle |
| `Hinge` | `HasRootBody` | RevoluteConnection joint body |
| `Slider` | `HasRootBody` | PrismaticConnection joint body |
| `Aperture` | `HasRootRegion` | Open passage region |
| `Container` | `HasCaseAsRootBody, HasStorageSpace, HasApertures, HasDoors, HasDrawers` | Cupboard/shelf with contents |

## Factory pattern

Most annotations are constructed via classmethods rather than direct `__init__`. Example:

```python
handle = Handle.create_with_new_body_in_world(
    name=PrefixedName("handle"), world=world,
    scale=Scale(0.1, 0.02, 0.02)
)
```

The factory creates a hollow box geometry via set algebra (`SimpleEvent`), builds `BoundingBoxCollection` from the event, attaches it to a new `Body`, and connects it to the world via `_create_with_connection_in_world`.

## Related

- **Uses:** [[sdt.world_description.world_entity.Body]], [[sdt.world_description.world_entity.Region]], [[sdt.world_description.connections]], [[sdt.world_description.shape_collection.ShapeCollection]]
- **Used by:** [[concept.semantic-annotation]], [[pycram.robot_plans.actions.core.misc]] (DetectAction — `IsPerceivable.class_label` for perception matching)

## Provenance

- `semantic_annotations.py:1-120` — concrete annotation classes, `Furniture`, `Handle` factory.
- `mixins.py:1-120` — all mixin traits with field definitions.
