---
id: giskardpy.motion_statechart.graph_node.Goal
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/graph_node.py
    lines: [899, 1080]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/goals/templates.py
    lines: [1, 65]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
used_by:
  - giskardpy.motion_statechart.goals
  - pycram.robot_plans.motions.gripper
  - pycram.robot_plans.motions.container
status: stable
tags: [giskardpy, motion-statechart, goal, composite, sequence, parallel, expand]
last_ingest: 2026-05-17
---

_Bundled page: `Goal` (composite MSC node), `EndMotion`, `CancelMotion`, `ThreadPayloadMonitor`, and the `Sequence`/`Parallel` template goals._

## `Goal` — composite node

```python
@dataclass(eq=False, repr=False)
class Goal(MotionStatechartNode):
    nodes: List[MotionStatechartNode]
    def expand(self, context): ...  # override to add children
    def add_node(self, node): ...   # register child with MSC + set parent_node
```

`Goal.expand(context)` is called during MSC compilation (before `build()`). Override it to instantiate child nodes and wire their conditions. Constraints:
- `EndMotion` may **not** be added as a direct child of a `Goal` (`EndMotionInGoalError`).
- A node may belong to only one parent Goal.

All pycram motion designators return a `Goal` subclass from `_motion_chart` — the Goal's `expand()` creates the concrete `Task` children.

## `EndMotion` — success termination

```python
class EndMotion(MotionStatechartNode):
    def build(context): return NodeArtifacts(observation=Scalar.const_true())
```

When `EndMotion` enters RUNNING, the MSC terminates successfully. Four factory methods wire the `start_condition`:

| Factory | Fires when |
|---|---|
| `EndMotion.when_true(node)` | `node.observation_variable` is TRUE |
| `EndMotion.when_false(node)` | `node.observation_variable` is FALSE |
| `EndMotion.when_all_true(nodes)` | all node observations TRUE |
| `EndMotion.when_any_true(nodes)` | any node observation TRUE |

`EndMotion` must be added at the top level of the MSC, not inside a `Goal`. This is enforced by `Goal._check_node_has_no_end_motion`.

## `CancelMotion` — failure termination

```python
class CancelMotion(MotionStatechartNode):
    exception: DataclassException  # raised when the node enters RUNNING
```

`on_tick` raises `exception` when the node is RUNNING. Used for error paths — if a safety monitor's condition fires, the MSC is cancelled with the given exception.

## `ThreadPayloadMonitor` — async observation

Abstract base; subclasses implement `_compute_observation() -> float`. Starts a daemon worker thread on `__post_init__`. `compute_observation()` signals the worker and returns the last cached result (UNKNOWN until first completion). Useful for expensive observations (IK reachability, collision queries) that can lag behind the control loop.

## Template goals

### `Sequence(Goal)`

Children start in order: child N starts when child N-1's observation is TRUE. Own observation = last child's observation.

```python
Sequence(nodes=[task_a, task_b, task_c])
# task_b.start_condition = task_a.observation_variable
# task_c.start_condition = task_b.observation_variable
# observation = task_c.observation_variable
```

### `Parallel(Goal)`

All children start together. Own observation = `minimum_success ≤ count(children with TRUE observation)`. Default `minimum_success = len(nodes)` (all must succeed).

## Related

- **Uses:** [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]]
- **Used by:** [[giskardpy.motion_statechart.goals]] (all concrete Goals extend this), pycram motion designators (`_motion_chart` returns a Goal)

## Open questions

- `CancelMotion.on_tick` raises the exception directly in the control loop — this breaks out of the tick loop. Whether the MSC's `cleanup()` call is guaranteed to run before the exception propagates depends on `MotionExecutor`'s exception handling, which needs verification.

## Provenance

- `graph_node.py:899-1080` — `Goal`, `EndMotion`, `CancelMotion`, `ThreadPayloadMonitor`.
- `goals/templates.py:1-65` — `Sequence`, `Parallel`.
