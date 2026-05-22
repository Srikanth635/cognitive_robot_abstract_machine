---
id: sdt.reasoning.WorldReasoner
kind: entity
package: sdt
source_paths:
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/world_reasoner.py
    lines: [1, 98]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/reasoner.py
    lines: [1, 100]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/predicates.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: semantic_digital_twin/src/semantic_digital_twin/reasoning/queries.py
    lines: [1, 80]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.world_description.world_entity.KinematicStructureEntity
  - sdt.reasoning.predicates
used_by:
  - concept.semantic-annotation
status: stable
tags: [sdt, reasoning, rdr, ripple-down-rules, predicate, eql, world-reasoner, case-reasoner]
last_ingest: 2026-05-17
---

_Bundled page: `WorldReasoner` (world-level), `CaseReasoner` (object-level), EQL predicates, and query helpers — the full SDT inference stack._

## `WorldReasoner`

```python
world_reasoner = WorldReasoner(world)
annotations: List[SemanticAnnotation] = world_reasoner.infer_semantic_annotations()
```

Wraps `CaseReasoner` for a whole `World`. Tracks `world_model_manager.version` and re-classifies only when the world topology has changed — otherwise returns the cached result.

`_update_world_attributes()` iterates the result dict and calls `setattr(world, attr_name, attr_value)` for every key except `"semantic_annotations"` — the RDR can update arbitrary world attributes alongside returning annotations.

**Fitting mode:** `fit_semantic_annotations(required_types, ...)` prompts an expert to extend the RDR for cases the current model cannot handle, enabling incremental knowledge acquisition without full retraining.

## `CaseReasoner`

Lower-level: reasons on any `case` object (not just a World). Uses `krrood.ripple_down_rules.GeneralRDR`, keyed by `case.__class__` in a class-level `CaseRDRs` dict (shared across all instances of the same case type).

RDR models are persisted to `<model_directory>/<ClassName.lower()>_rdr.*`. The first `CaseReasoner` constructed for a new case type initialises the `GeneralRDR` and loads the model from disk.

```python
reasoner = CaseReasoner(case)
result: Dict[str, Any] = reasoner.reason()          # classification
reasoner.fit_attribute("attr_name", [types], False)  # interactive fitting
```

## EQL predicates (`predicates.py`)

Decorated with `@symbolic_function` — usable both as concrete Python functions and as lazy EQL expressions when called with `Variable` arguments. Now covered in full by the dedicated page [[sdt.reasoning.predicates]]; summary below.

| Predicate | Signature | Notes |
|---|---|---|
| `stable(obj)` | `Body → bool` | Raises `NotImplementedError` (needs multiverse) — dead branch |
| `contact(body1, body2, threshold)` | `Body, Body, float → bool` | FCL distance < threshold |
| `is_supported_by(obj, surface)` | `Body, HasSupportingSurface → bool` | Spatial containment test |
| `is_supporting(surface, obj)` | inverse of above | |
| `visible(camera, obj)` | `Camera, Body → bool` | Segmentation mask; ignores camera orientation |
| `reachable(pose, root, tip)` | | Delegates to `world.compute_inverse_kinematics()` |
| `compute_euclidean_planar_distance(b1, b2, dim)` | | XY-plane distance |

Spatial relation classes (`LeftOf`, `RightOf`, `Above`, `Below`, `Behind`, `InFrontOf`) and
placement helpers (`is_place_occupied`, `InsideOf`, `is_body_in_region`) are also in this module.
See [[sdt.reasoning.predicates]] for the full 615-line coverage.

## Query helpers (`queries.py`)

Higher-level EQL query builders returning krrood `Entity` descriptors for use in `underspecified(...)` plans:

| Function | Returns |
|---|---|
| `semantic_annotations_on_surfaces(surfaces, world)` | `List[HasRootBody]` |
| `get_next_object_using_planar_distance(body, surface, dim)` | `Entity[SemanticAnnotation]` ordered by distance |
| `goal_surface_of_object(obj, surfaces, threshold)` | Most similar `HasSupportingSurface` by class inheritance distance |

## Related

- **Uses:** [[sdt.world.World]], [[sdt.world_description.world_entity.KinematicStructureEntity]], [[sdt.reasoning.predicates]]
- **Used by:** [[concept.semantic-annotation]]

## Open questions

- The RDR model for world-level reasoning (`world_rdr/`) is loaded from `dirname(__file__)` — modifying model files at runtime risks race conditions if multiple processes share the same directory.
- `stable()` raises `NotImplementedError` unconditionally — any plan that calls it will fail until a multiverse physics backend is integrated.

## Provenance

- `world_reasoner.py:1-98` — `WorldReasoner`, `infer_semantic_annotations`, `reason`, `fit_semantic_annotations`.
- `reasoner.py:1-100` — `CaseReasoner`, `CaseRDRs`, `reason`, `fit_attribute`.
- `predicates.py:1-80` — partial coverage bundled here; see [[sdt.reasoning.predicates]] for full 615-line ingest.
- `queries.py:1-80` — `semantic_annotations_on_surfaces`, `get_next_object_using_planar_distance`, `goal_surface_of_object`.
