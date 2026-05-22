---
id: pycram.robot_plans.motions.misc
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/misc.py
    lines: [1, 26]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.robot_plans.BaseMotion
  - pycram.perception.PerceptionQuery
used_by: []
fields:
  query:
    type: pycram.perception.PerceptionQuery
    description: Query describing what to detect and within which bounding box.
status: stable
tags: [motion, perception, detect]
last_ingest: 2026-05-19
---

_One `BaseMotion` subclass for perception: `DetectingMotion` wraps a [[pycram.perception.PerceptionQuery]] and produces no giskardpy motion — `_motion_chart` returns `None`._

## DetectingMotion

| Field | Type | Description |
|-------|------|-------------|
| `query` | `PerceptionQuery` | Query describing what and where to detect |

`_motion_chart` is `pass` (returns `None`). This is the only motion in the codebase whose chart yields no giskardpy Task.

The actual perception call happens in `DetectAction.execute()` (see [[pycram.robot_plans.actions.core.misc]]), not in the motion. `DetectingMotion` exists as a designator container and plan-graph leaf but does not drive the QP controller.

## Open questions

- Since `_motion_chart` returns `None`, `MotionExecutor` would receive a `None` task. It's unclear how the executor handles a `None` in the `motions` list. May be that `DetectingMotion` is never actually added to a MotionNode — `DetectAction` calls `PerceptionQuery.from_world()` directly in `execute()`.

## Related

**Uses:** [[pycram.robot_plans.BaseMotion]], [[pycram.perception.PerceptionQuery]]

**Used by:** (none confirmed — `DetectAction.execute()` calls `PerceptionQuery.from_world()` directly and does not instantiate `DetectingMotion`)

## Provenance

- `pycram/src/pycram/robot_plans/motions/misc.py` lines 1–26 (commit `0528d8cf3`) — `DetectingMotion`.
