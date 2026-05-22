---
id: concept.semantic-annotation
kind: concept
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/world_description/world_entity.py
    lines: [555, 650]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/semantic_annotations/mixins.py
    lines: [1, 120]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/reasoner.py
    lines: [1, 100]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/world_reasoner.py
    lines: [1, 98]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.semantic_annotations.SemanticAnnotations
  - sdt.reasoning.WorldReasoner
used_by:
  - concept.world
  - bridge.pycram-sdt
status: stable
tags: [sdt, semantic, annotation, rdr, reasoning, eql, predicate, furniture, robot]
last_ingest: 2026-05-17
---

_The SDT semantic layer: `SemanticAnnotation` groups `KinematicStructureEntity` nodes into named, typed interpretations of the world; `WorldReasoner` uses Ripple Down Rules to infer them automatically._

## Core idea

Physical bodies and regions in the world are semantically neutral — a `Body` is just geometry. `SemanticAnnotation` overlays meaning: a `Table` is a `Body` with a supporting surface; a `Drawer` is a `Body` + a `Region` + a `PrismaticConnection`. Annotations carry references to their constituent bodies and are stored in `world.semantic_annotations`.

A single `Body` can participate in multiple annotations simultaneously. Annotations are therefore not a partition of the world — they are an open overlay.

## `SemanticAnnotation` base class

```python
@dataclass(eq=False)
class SemanticAnnotation(WorldEntityWithSimulatorProperties): ...
```

Key design choices:
- **Hash = type + sorted KSE IDs.** `hash(ann) = hash(tuple([type(ann)] + sorted([kse.id for kse in ann.kinematic_structure_entities])))`. Two annotation instances with the same type and same bodies are equal, regardless of object identity. This is required for rule-based equality checks in the RDR.
- **`kinematic_structure_entities` is recursive.** The property collects all `KinematicStructureEntity` objects reachable by recursively traversing annotation fields — including nested annotations (e.g. a `Container` annotation that holds a `StorageSpace` annotation).
- **`_synonyms: ClassVar[Set[str]]`** — alternative class names accepted by string-based lookup (e.g. `"cup"` matches `MugAnnotation` if registered as a synonym).
- **`class_name_tokens()`** — splits the CamelCase class name into lowercase tokens for approximate text matching.

## Mixin architecture

`semantic_annotations/mixins.py` defines compositional mixin traits that annotations combine:

| Mixin | Adds |
|---|---|
| `HasRootBody` | `root: Body` — primary rigid body of the annotation |
| `HasRootRegion` | `root: Region` — primary spatial volume |
| `HasSupportingSurface` | Surface area + `objects: List[...]]` — inferred objects on top |
| `HasStorageSpace` | Interior `Region` for containment queries |
| `HasDrawers` | `List[Drawer]` sub-annotations |
| `HasDoors` | `List[Door]` sub-annotations with `RevoluteConnection` |
| `HasHandle` | Reference to a `Handle` annotation |
| `IsPerceivable` | `class_label: Optional[str]` for perception matching |

Concrete annotations in `semantic_annotations.py` compose these traits. Example: `Drawer(HasRootBody, HasHandle, HasStorageSpace)`.

## Reasoning via Ripple Down Rules

`WorldReasoner` wraps a `CaseReasoner` (from krrood's `GeneralRDR`) to infer which annotations a world contains:

```python
world_reasoner = WorldReasoner(world)
annotations = world_reasoner.infer_semantic_annotations()
# Returns List[SemanticAnnotation] by classifying world.bodies + world.connections
```

The RDR model is stored on disk (`.pkl` / `.json`) in `reasoning/` and loaded lazily per case type. `WorldReasoner.reason()` re-classifies only when `world_model_manager.version` has changed — it caches results across ticks.

**Fitting mode** allows incremental knowledge gain: a human expert is prompted to provide annotation rules for cases the current model does not handle, extending the rule tree without retraining from scratch.

## EQL predicate integration

Reasoning predicates in `reasoning/predicates.py` are decorated with `@symbolic_function`, making them composable with krrood EQL:

- `stable(obj: Body)` — physics-stability check (requires multiverse)
- `contact(body1, body2, threshold)` — FCL collision distance check
- `is_supported_by(obj, surface)` — tests if `obj` lies on `surface`'s supporting region

These are used in `ActionDescription.pre_condition` / `post_condition` to gate action execution.

## Related

- **Uses:** [[sdt.world_description.world_entity.KinematicStructureEntity]], [[sdt.semantic_annotations.SemanticAnnotations]], [[sdt.reasoning.WorldReasoner]]
- **Used by:** [[concept.world]], [[bridge.pycram-sdt]]

## Open questions

- `WorldReasoner._update_world_attributes` calls `setattr(world, attr_name, attr_value)` for every key in the RDR result except `"semantic_annotations"` — this means the RDR can modify arbitrary World attributes. The full set of world attributes the RDR is allowed to set is not documented here.
