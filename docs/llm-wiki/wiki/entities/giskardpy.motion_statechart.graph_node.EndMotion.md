---
id: giskardpy.motion_statechart.graph_node.EndMotion
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/graph_node.py
    lines: [899, 1080]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
used_by:
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - pycram.motion_executor.MotionExecutor
  - pycram.locations.locations.GiskardLocation
status: stub
tags: [giskardpy, motion-statechart, termination, end-motion]
last_ingest: 2026-05-18
---

_Stub. `EndMotion` is a `MotionStatechartNode` subclass that signals successful completion of a motion. When its lifecycle state reaches RUNNING the MSC terminates successfully. Documented in [[giskardpy.motion_statechart.graph_node.Goal]] (bundled page: Goal, EndMotion, CancelMotion)._

To be expanded on the next ingest that touches `giskardpy.motion_statechart.graph_node`.
