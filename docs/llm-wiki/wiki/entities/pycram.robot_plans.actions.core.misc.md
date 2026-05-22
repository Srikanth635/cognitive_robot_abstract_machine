---
id: pycram.robot_plans.actions.core.misc
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/actions/core/misc.py
    lines: [1, 79]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.ActionDescription
  - pycram.perception.PerceptionQuery
  - sdt.world_description.world_entity.Region
  - sdt.semantic_annotations.SemanticAnnotations
used_by:
  - pycram.robot_plans.actions.composite
fields:
  technique:
    type: str
    description: Detection strategy (DetectionTechnique enum — TYPES, ALL, etc.); not yet a separate wiki page.
  state:
    type: str
    description: Start/stop signal for continuous detection (DetectionState enum); optional, may be None.
  object_sem_annotation:
    type: sdt.semantic_annotations.SemanticAnnotations
    description: SDT semantic annotation type to search for; optional — defaults to SemanticEnvironmentAnnotation (detect everything).
  region:
    type: sdt.world_description.world_entity.Region
    description: Restrict detection to a specific SDT Region; optional — if None uses a hardcoded 4×4×3 m bounding box around the robot root.
status: stable
tags: [action, detect, perception, region]
last_ingest: 2026-05-19
---

_One `ActionDescription` subclass for object detection: `DetectAction` constructs a [[pycram.perception.PerceptionQuery]] from a `DetectionTechnique` and optional region, then calls `query.from_world()` — bypassing the giskardpy MSC entirely._

## DetectAction

| Field | Type | Description |
|-------|------|-------------|
| `technique` | `DetectionTechnique` | Detection strategy (e.g. `TYPES`, `ALL`) |
| `state` | `DetectionState \| None` | Start/stop signal for continuous detection; optional |
| `object_sem_annotation` | `Type[SemanticAnnotation] \| None` | Target semantic annotation type to search for |
| `region` | `Region \| None` | Restrict search to a specific SDT `Region`; optional |

**execute()** logic:
1. If `region` is provided, computes a `BoundingBox` from `region.area.as_bounding_box_collection_in_frame(robot.root).bounding_box`.
2. If no region, uses a hardcoded default bounding box: `x ∈ [-1, 3], y ∈ [-1, 3], z ∈ [0, 3]` relative to the robot root.
3. If `object_sem_annotation` is `None`, falls back to `SemanticEnvironmentAnnotation` (detect everything).
4. Constructs `PerceptionQuery(annotation_type, bounding_box, robot, world)` and calls `query.from_world()`.

**No giskardpy MSC involved.** Detection results are returned by `from_world()` and presumably added to the world's semantic annotations (consumed by the caller via `world.semantic_annotations` diff — as done in `SearchAction`).

`validate()` is a no-op (commented-out `PerceptionObjectNotFound` raise is disabled).

## Related

**Uses:** [[pycram.robot_plans.ActionDescription]], [[pycram.perception.PerceptionQuery]], [[sdt.world_description.world_entity.Region]]

**Used by:** [[pycram.robot_plans.actions.composite]]

## Open questions

- The hardcoded default bounding box (`x: -1 to 3, y: -1 to 3, z: 0 to 3`) is relative to the robot root. For large rooms or distant objects this may miss detections. No config parameter for it.
- `PerceptionQuery.from_world()` return value is not used or stored in `execute()`. How detected objects become available to the plan (via `world.semantic_annotations`) is not explicit in this file.

## Provenance

- `pycram/src/pycram/robot_plans/actions/core/misc.py` lines 1–79 (commit `0528d8cf3`) — `DetectAction`.
