---
id: pycram.perception.PerceptionQuery
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/perception.py
    lines: [1, 77]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - sdt.world.World
  - sdt.robots.abstract_robot.AbstractRobot
  - sdt.world_description.geometry.BoundingBox
  - sdt.reasoning.predicates
used_by:
  - pycram.robot_plans.actions.core.misc
  - pycram.robot_plans.motions.misc
status: stable
tags: [perception, query, detection, visibility, semantic-annotation, region-filter]
last_ingest: 2026-05-17
---

_Dataclass encapsulating a perception request: queries **existing** `SemanticAnnotation` instances in the SDT world, filters by spatial region and camera visibility, and returns matching `Body` objects._

## Purpose

`PerceptionQuery` is the interface between pycram's action layer and SDT's semantic world
state. It does **not** run a detector, dispatch to a sensor, or populate new annotations.
It reads annotations already present in `world.semantic_annotations` and returns which tagged
bodies are (a) inside the requested region and (b) visible from the robot's camera.

`from_robokudo()` exists as a stub for future real-robot sensor integration but is currently
empty (`pass`).

## Construction

```python
query = PerceptionQuery(
    semantic_annotation=CupAnnotation,     # Type[SemanticAnnotation] to search for
    region=bounding_box,                   # BoundingBox: spatial filter in world frame
    robot=ctx.robot,                       # AbstractRobot — supplies Camera sensor
    world=ctx.world,                       # SDT World to query
)
result: List[Body] = query.from_world()
```

## `from_world()` algorithm

```python
def from_world(self) -> List[Body]:
```

1. `world.get_semantic_annotations_by_type(semantic_annotation)` — retrieves all existing
   instances of the requested annotation type from the world.
2. Collect all `.bodies` across every annotation instance.
3. Spatial filter: `region.contains(body.global_transform.to_position())` — keeps only bodies
   whose global position falls within the `BoundingBox`.
4. Visibility filter: `visible(robot_camera, body)` — uses `sdt.reasoning.predicates.visible()`
   to check camera line-of-sight. Camera = first `Camera` sensor found in `robot.sensors`.
5. Returns `List[Body]` of bodies passing both filters.

## Key insight: reads, does not write

`from_world()` makes **no mutations** to world state. The question of "how does
`PerceptionQuery.from_world()` populate `world.semantic_annotations`" reflects a wrong
assumption — population is handled upstream by a vision system or synthetic world setup
before `PerceptionQuery` is called. `PerceptionQuery` is purely a query over an already-
populated world.

## `from_robokudo()`

Empty stub (`pass`). Intended as the real-robot path where a RoboKudo query is sent over
ROS 2 and results are integrated into the world. Not yet implemented at this commit.

## Related

**Uses:** [[sdt.world.World]], [[sdt.robots.abstract_robot.AbstractRobot]], [[sdt.world_description.geometry.BoundingBox]], [[sdt.reasoning.predicates]]

**Used by:** [[pycram.robot_plans.actions.core.misc]], [[pycram.robot_plans.motions.misc]]

**See also:** [[sdt.semantic_annotations.SemanticAnnotations]], [[sdt.reasoning.WorldReasoner]]

## Open questions

- `robot.sensors` is filtered for `Camera` instances and only the **first** match is used.
  Robots with multiple cameras (e.g. head + wrist) always perceive from the first camera.
  No API to select which camera to query.
- Whether `world.get_semantic_annotations_by_type` searches a snapshot or the live (mutable)
  world is unknown — `PerceptionQuery` does not deepcopy the world, so concurrent world
  mutations during `from_world()` could produce inconsistent results.

## Provenance

- `pycram/src/pycram/perception.py:1-77` — `PerceptionQuery` dataclass, `from_world()`, stub
  `from_robokudo()`.
