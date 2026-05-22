---
id: giskardpy.motion_statechart.motion_statechart.MotionStatechart
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/motion_statechart.py
    lines: [276, 540]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
  - giskardpy.motion_statechart.graph_node.Task
  - giskardpy.motion_statechart.graph_node.EndMotion
used_by:
  - bridge.sdt-giskardpy
  - concept.motion-statechart
  - giskardpy.qp.qp_controller
  - pycram.motion_executor.MotionExecutor
status: stable
tags: [giskardpy, motion-statechart, lifecycle, observation, casadi, qp, rustworkx]
last_ingest: 2026-05-17
---

_The giskardpy Motion State Chart: a compiled `rustworkx.PyDiGraph` of nodes with trinary lifecycle/observation states, driving QP constraint selection each control cycle._

## Purpose

`MotionStatechart` is the central orchestration object in giskardpy. It is built at runtime
(by `MotionExecutor`), populated with `Task` and `Goal` nodes from pycram motion designators,
compiled once, then ticked in a control loop. On each tick, it evaluates compiled CasADi
functions to update which nodes are RUNNING, and merges their constraint collections for the
QP solver.

## Key attributes

| Name | Kind | Notes |
|---|---|---|
| `rx_graph` | `rx.PyDiGraph[MotionStatechartNode]` | Underlying directed multigraph. Not a DAG — edges represent condition dependencies, not tree structure. |
| `observation_state` | `ObservationState` | Numpy array (one float per node) tracking trinary observation. Backed by a compiled CasADi function. |
| `life_cycle_state` | `LifeCycleState` | Numpy array tracking `NOT_STARTED/RUNNING/PAUSED/DONE`. Backed by a compiled CasADi state machine. |
| `history` | `StateHistory` | Appends `StateHistoryItem` only when state changes; used for Gantt chart plotting and debugging. |

## Key methods

| Name | Notes |
|---|---|
| `add_node(node)` | Adds node to graph, sets `node.index`, calls `_post_add_to_motion_statechart()`, grows state arrays. |
| `compile(context)` | One-time setup: expand goals → build nodes → add transition edges → compile CasADi functions. Must be called before `tick()`. |
| `tick(context)` | One control cycle: update observation state (compiled fn + `on_tick` callbacks) → update lifecycle state (compiled fn + lifecycle callbacks). |
| `combine_constraint_collections_of_nodes()` | Merges all nodes' `_constraint_collection` objects by prefixing with `node.unique_name`. Returns a single `ConstraintCollection` for the QP controller. |
| `get_nodes_by_type(type_)` | Filters `self.nodes` by type — used to find all `Goal` nodes in `_expand_goals()`. |
| `is_end_motion()` | True if any `EndMotion` node has lifecycle state `RUNNING`. Signals the control loop to stop. |

## Compilation detail

`compile(context)` in order:
1. **expand goals** — `goal.expand(context)` instantiates child nodes; child `Task` nodes added via `add_node`.
2. **build nodes** — `node.build(context)` → `NodeArtifacts`; assigns `_constraint_collection` and `_observation_expression` on each node.
3. **add transitions** — adds directed edges (`TrinaryCondition`) from each node to its condition dependencies in the graph.
4. **compile observation** — CasADi `Vector` of per-node `if_eq_cases` expressions compiled with `[observation, lifecycle, world.state, float_variables]` as parameters; args bound to numpy memory views.
5. **compile lifecycle** — CasADi `Vector` of per-node four-case state machine compiled with `[observation, lifecycle]`; args bound to numpy memory views.

After compilation, `tick()` calls `.evaluate()` on the compiled functions + `np.copyto` — zero Python overhead per node.

## Related

- Node base: [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- Task node: [[giskardpy.motion_statechart.graph_node.Task]]
- Concept: [[concept.motion-statechart]]
- Package: [[giskardpy]]
- MSC executor: [[pycram.motion_executor.MotionExecutor]]

## Open questions

- `combine_constraint_collections_of_nodes()` iterates all nodes regardless of lifecycle state.
  Whether `link_to_motion_statechart_node` adds lifecycle gating to the constraints automatically
  (so inactive Tasks produce zero constraints) is not yet confirmed — needs QP layer ingest.

## Provenance

- `giskardpy/src/giskardpy/motion_statechart/motion_statechart.py:276-540` at commit `0528d8cf3` —
  `MotionStatechart` class: constructor, `compile()`, `tick()`, `combine_constraint_collections_of_nodes()`.
- Same file, lines 39–270 — `State`, `LifeCycleState`, `ObservationState`, `StateHistory`.
