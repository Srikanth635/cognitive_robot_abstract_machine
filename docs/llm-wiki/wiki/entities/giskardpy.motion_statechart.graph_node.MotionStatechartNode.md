---
id: giskardpy.motion_statechart.graph_node.MotionStatechartNode
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/graph_node.py
    lines: [362, 878]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.TrinaryCondition
  - giskardpy.motion_statechart.graph_node.NodeArtifacts
used_by:
  - giskardpy.motion_statechart.graph_node.Task
  - giskardpy.motion_statechart.graph_node.Goal
  - giskardpy.motion_statechart.graph_node.EndMotion
  - giskardpy.motion_statechart.graph_node.CancelMotion
  - giskardpy.motion_statechart.motion_statechart.MotionStatechart
  - concept.motion-statechart
  - giskardpy.motion_statechart.goals
  - giskardpy.motion_statechart.monitors
  - giskardpy.motion_statechart.tasks
  - giskardpy.qp.constraint_collection.ConstraintCollection
status: stable
tags: [giskardpy, motion-statechart, lifecycle, conditions, trinary, abstract]
last_ingest: 2026-05-18
---

_Abstract base for every node in a giskardpy Motion State Chart; owns four `TrinaryCondition` lifecycle transitions and produces `NodeArtifacts` from `build()`._

## Purpose

`MotionStatechartNode` is the single base class for all MSC graph nodes: `Task` (leaf QP constraints), `Goal` (composites that expand into children), `EndMotion` / `CancelMotion` (termination signals), and monitor nodes. The lifecycle state machine and condition-wiring logic are entirely in this class.

## Lifecycle state machine

Each node has four `TrinaryCondition` fields, one per `TransitionKind`:

| Condition | Default | Semantics |
|---|---|---|
| `start_condition` | `const_true` | NOT_STARTED → RUNNING |
| `pause_condition` | `const_false` | RUNNING ↔ PAUSED |
| `end_condition` | `const_false` | RUNNING/PAUSED → DONE |
| `reset_condition` | `const_false` | Any → NOT_STARTED |

`LifeCycleValues`: `NOT_STARTED=0`, `RUNNING=1`, `PAUSED=2`, `DONE=3`, `FAILED=4`.

`create_lifecycle_transitions()` assembles four symbolic `if_cases` expressions — one per state — that read ancestor conditions via `_create_any_ancestor_condition_true()`. The ancestor OR-chain means: if any ancestor's end/reset condition fires, all descendants transition immediately.

## Two-phase initialisation

1. **`__post_init__`** — initialises the four `TrinaryCondition` fields with defaults.
2. **`_post_add_to_motion_statechart()`** — called after insertion into an MSC; creates `ObservationVariable` and `LifeCycleVariable` (both require knowing `self.index`).

Accessing `life_cycle_variable`, `observation_variable`, or `motion_statechart` before step 2 raises `NotInMotionStatechartError`.

## Build and tick hooks

| Method | When called | Returns |
|---|---|---|
| `build(context)` | Once, during MSC compilation | `NodeArtifacts` (constraints + observation expr + debug exprs) |
| `on_tick(context)` | Every tick while RUNNING | `Optional[ObservationStateValues]` — overrides `NodeArtifacts.observation` if set |
| `on_start(context)` | On NOT_STARTED → RUNNING | — |
| `on_pause(context)` | On RUNNING → PAUSED | — |
| `on_unpause(context)` | On PAUSED → RUNNING | — |
| `on_end(context)` | On RUNNING/PAUSED → DONE | — |
| `on_reset(context)` | On any → NOT_STARTED | — |
| `cleanup(context)` | After EndMotion / CancelMotion | — |

**Warning documented in source:** all `on_*` and `on_tick` methods are called inside the control loop — they must be fast.

## `TrinaryCondition` invariants

`update_expression()` enforces three sanity checks:
1. Expression must be a `Scalar`.
2. A `start_condition` must not reference its own `ObservationVariable` (would be a self-loop).
3. All free variables must be `ObservationVariable` instances — lifecycle conditions cannot reference raw CasADi symbols.

## Key properties

| Property | Returns |
|---|---|
| `start_condition` / `pause_condition` / `end_condition` / `reset_condition` | `Scalar` (the condition expression) |
| `life_cycle_state` | `LifeCycleValues` (current state, read from MSC) |
| `observation_state` | `float` (trinary: FALSE/UNKNOWN/TRUE) |
| `depth` | Walk-to-root edge count |
| `unique_name` | `"{name}#{index}"` |

## Subclasses

See [[giskardpy.motion_statechart.graph_node.Goal]], [[giskardpy.motion_statechart.graph_node.EndMotion]], [[giskardpy.motion_statechart.graph_node.CancelMotion]], [[giskardpy.motion_statechart.graph_node.Task]].

## Related

- **Uses:** [[giskardpy.motion_statechart.graph_node.TrinaryCondition]], [[giskardpy.motion_statechart.graph_node.NodeArtifacts]]
- **Used by:** [[giskardpy.motion_statechart.motion_statechart.MotionStatechart]], [[concept.motion-statechart]]

## Provenance

- `graph_node.py:362-878` — class definition, `__post_init__`, `_post_add_to_motion_statechart`, `create_lifecycle_transitions`, `_create_*_transitions`, `build`, all `on_*` hooks, condition properties and setters.
