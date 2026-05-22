---
id: giskardpy
kind: package
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/__init__.py
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by: []
status: stable
tags: [giskardpy, package, motion-statechart, qp, robot-control]
last_ingest: 2026-05-17
---

_giskardpy (v1.0.0): a reactive motion controller for robots using a Motion State Chart (MSC) to orchestrate QP-based whole-body control._

## Purpose

giskardpy provides whole-body robot motion control by combining two layers:

1. **Motion State Chart (MSC)** — a directed graph of `MotionStatechartNode` objects that coordinates
   which motion tasks are active, paused, or complete using trinary lifecycle conditions compiled into
   efficient CasADi functions.
2. **QP (Quadratic Programming) solver** — converts the constraint collections produced by active
   `Task` nodes into a quadratic program solved each control cycle to compute joint velocities.

pycram's motion designators produce `Task` nodes whose constraints feed into the MSC/QP pipeline.
See [[bridge.pycram-giskardpy]].

## Subpackage tour

| Subpackage | Role |
|---|---|
| `motion_statechart` | [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]] — MSC graph, compile/tick loop, lifecycle machinery. Contains `goals/`, `tasks/`, `monitors/` subdirectories. |
| `qp` | QP controller: `ConstraintCollection`, `QPDataFactory`, `QPController`, `QPSolver` (in `solvers/`). Converts active Task constraints into joint velocity commands. |
| `model` | Robot/kinematic model used by the QP layer for FK and Jacobian computation. |
| `tree` | Behavior/task tree structures (likely integrates with the MSC lifecycle). |
| `data_types` | Core data structures (joint states, constraint data, etc.). |
| `middleware` | Middleware abstraction (ROS/non-ROS execution). |
| `ros2_tools` | ROS 2 integration utilities. |
| `utils` | Shared utilities. |

## Execution flow

```
pycram ActionNode._perform()
  → collect_motions()              # harvests MotionNode tasks
  → MotionExecutor.execute_msc()   # builds MotionStatechart, calls compile()
       → MSC.compile(context)      # expand Goals → build nodes → add transitions → compile CasADi
  → control loop: MSC.tick()       # update observation + lifecycle states each cycle
       → MSC.combine_constraints() # merge Task ConstraintCollections
  → QPController.solve()           # compute joint velocities
```

## Core concept

See [[concept.motion-statechart]] for the conceptual architecture of how MSC nodes, lifecycle
conditions, and observation states interact. See [[giskardpy.motion_statechart.graph_node.Task]]
for the node type that pycram's motion designators produce.

## Related

- Core concept: [[concept.motion-statechart]]
- MSC class: [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]]
- Task node: [[giskardpy.motion_statechart.graph_node.Task]]
- pycram interface: [[bridge.pycram-giskardpy]]

## Provenance

- `giskardpy/src/giskardpy/__init__.py` at commit `0528d8cf3` — version 1.0.0.
- `giskardpy/src/giskardpy/motion_statechart/motion_statechart.py` at same commit — `MotionStatechart` class.
