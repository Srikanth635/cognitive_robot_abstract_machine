---
id: concept.motion-statechart
kind: concept
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/motion_statechart.py
    lines: [276, 540]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/graph_node.py
    lines: [362, 897]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - concept.forward-kinematics
  - giskardpy.motion_statechart.monitors
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - giskardpy.motion_statechart.graph_node.Task
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
  - giskardpy.motion_statechart.graph_node.TrinaryCondition
used_by:
  - concept.qp-controller
  - bridge.pycram-giskardpy
status: stable
tags: [concept, giskardpy, motion-statechart, lifecycle, trinary, qp, casadi]
last_ingest: 2026-05-17
---

_A directed graph of nodes with compiled trinary lifecycle transitions and observation states, driving which QP constraints are active each control cycle._

## What the Motion State Chart is

A Motion State Chart (MSC) is a `rustworkx.PyDiGraph` of `MotionStatechartNode` objects. Each node
simultaneously tracks two states:

- **Lifecycle state** — one of four values: `NOT_STARTED`, `RUNNING`, `PAUSED`, `DONE`. Only
  `RUNNING` nodes contribute to QP constraint solving.
- **Observation state** — trinary: `TrinaryTrue` / `TrinaryFalse` / `TrinaryUnknown`. Represents
  what the node "observes" about the world (e.g., "has the gripper reached the goal pose?").

## Lifecycle transitions

Each node has four CasADi symbolic conditions controlling its lifecycle:

| Condition | Transition | Default |
|---|---|---|
| `start_condition` | NOT_STARTED → RUNNING | `const_true` (starts immediately) |
| `pause_condition` | RUNNING ↔ PAUSED | `const_false` (never pauses) |
| `end_condition` | RUNNING/PAUSED → DONE | `const_false` (never ends on its own) |
| `reset_condition` | any → NOT_STARTED | `const_false` |

Priority when multiple conditions fire simultaneously: **reset > end > pause > start**.

Parent nodes' conditions propagate to children: a parent's `end_condition` firing also ends its
children. This enables `Goal` nodes to act as lifecycle scopes for their `Task` children.

## Node type hierarchy

```
MotionStatechartNode
├── Goal          — container; expand() instantiates child nodes at compile time
├── Task          — leaf; build() produces ConstraintCollection for QP
├── EndMotion     — signals motion complete; observation = const_true
├── CancelMotion  — signals motion cancelled; raises on tick
└── ThreadPayloadMonitor — async observer; returns TrinaryUnknown until first result
```

- **`Goal.expand(context)`** is called once at `compile()` time to create its child nodes. Child
  `Task` nodes are added to the MSC graph and linked under the Goal.
- **`Task.build(context)`** returns a `NodeArtifacts` containing a `ConstraintCollection` — the
  QP constraints this task enforces. Also sets an optional `observation` expression.
- **`EndMotion.when_true(node)`** and **`when_false(node)`** are factory methods that wire an
  `EndMotion` node to fire when another node's observation variable reaches true/false.

## Compilation (called once before the control loop)

```python
msc.compile(context)
```

Steps in order:
1. `sanity_check()` — validates MSC is non-empty and well-formed.
2. `_expand_goals(context)` — calls `goal.expand()` for every `Goal` node; child nodes are added to the MSC.
3. `_build_nodes(context)` — calls `node.build(context)` on each node; extracts `NodeArtifacts`:
   assigns `node._constraint_collection` and `node._observation_expression`.
4. `_add_transitions()` — adds directed edges (`TrinaryCondition` objects) between nodes based on condition dependencies.
5. `observation_state.compile(context)` — compiles a CasADi function parameterised by
   `[observation, lifecycle, world.state, float_variables]`.
6. `life_cycle_state.compile()` — compiles a CasADi state-machine function parameterised by
   `[observation, lifecycle]` using `if_eq_cases` for each node's four transitions.

Both compiled functions are bound via `bind_args_to_memory_view` to numpy array slices, so
`tick()` requires no Python-level data movement.

## The control loop (tick)

```python
while not msc.is_end_motion():
    msc.tick(context)
    constraints = msc.combine_constraint_collections_of_nodes()
    velocities = qp_controller.solve(constraints)
    robot.apply_velocities(velocities)
```

Each `tick()`:
1. **Update observation state** — runs the compiled CasADi observation function; then calls
   `node.on_tick(context)` for every `RUNNING` node (overrides compiled observation if non-None).
2. **Update lifecycle state** — runs the compiled CasADi state-machine function; triggers
   `on_start`, `on_pause`, `on_unpause`, `on_end`, `on_reset` callbacks for changed nodes.

`combine_constraint_collections_of_nodes()` merges the `ConstraintCollection` of every node
(whether RUNNING or not — the constraint must self-gate on lifecycle state internally, or the
`link_to_motion_statechart_node` call does so automatically).

## How pycram uses the MSC

1. `BaseMotion._motion_chart` returns a `Task` (or a `Goal` containing tasks).
2. `ActionNode.collect_motions()` harvests all descendant motion tasks.
3. `MotionExecutor` assembles them into a `MotionStatechart`, calls `compile()`, and runs the
   control loop until `is_end_motion()`.
4. The actual joint velocity computation happens in the QP layer — giskardpy internal, not yet
   ingested.

## Related

- MSC class: [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]]
- Node base: [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- Task node: [[giskardpy.motion_statechart.graph_node.Task]]
- Package: [[giskardpy]]
- pycram interface: [[bridge.pycram-giskardpy]]

## Resolved notes

- **Lifecycle gating (resolved):** `combine_constraint_collections_of_nodes()` iterates all nodes, not only RUNNING nodes. Gating is automatic: `ConstraintCollection.link_to_motion_statechart_node(node)` multiplies each constraint's `quadratic_weight` by `is_running = if_eq(lifecycle_var, RUNNING, 1, 0)`. Inactive constraints have zero weight; the QP matrix dimensions stay fixed each tick.
- **QP controller internals (resolved):** See [[concept.qp-controller]], [[giskardpy.qp.qp_controller]].

## Provenance

- `giskardpy/src/giskardpy/motion_statechart/motion_statechart.py:276-540` at commit `0528d8cf3` —
  `MotionStatechart`, `LifeCycleState.compile()`, `ObservationState.compile()`, `tick()`.
- `giskardpy/src/giskardpy/motion_statechart/graph_node.py:362-897` at commit `0528d8cf3` —
  `MotionStatechartNode` lifecycle machinery, `Task`, `Goal`, `EndMotion` class definitions.
