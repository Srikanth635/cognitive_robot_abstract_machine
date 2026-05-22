---
id: giskardpy.motion_statechart.graph_node.TrinaryCondition
kind: entity
package: giskardpy
source_paths:
  - path: giskardpy/src/giskardpy/motion_statechart/graph_node.py
    lines: [64, 263]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: giskardpy/src/giskardpy/motion_statechart/data_types.py
    lines: [1, 54]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - giskardpy.motion_statechart.graph_node.MotionStatechartNode
  - concept.motion-statechart
status: stable
tags: [giskardpy, trinary, condition, lifecycle, transition, scalar]
last_ingest: 2026-05-17
---

_Wrapper around a CasADi `Scalar` expression representing a transition guard in the MSC lifecycle; three possible values: TRUE / FALSE / UNKNOWN._

## Purpose

`TrinaryCondition` is the condition type that drives all four lifecycle transitions (start / pause / end / reset) of a `MotionStatechartNode`. Its `expression` is a symbolic `Scalar` composed of `ObservationVariable` references via trinary logic operators (`trinary_logic_and`, `trinary_logic_or`, `trinary_logic_not`). This means transition guards can depend on the runtime observation states of other nodes.

## Trinary logic values

| Constant | Numeric value | Semantics |
|---|---|---|
| `Scalar.const_true()` | 1.0 | Condition satisfied |
| `Scalar.const_false()` | 0.0 | Condition not satisfied |
| `Scalar.const_trinary_unknown()` | 0.5 | Not yet determined |

`ObservationStateValues` mirrors these as a `FloatEnum`: `TRUE=1.0`, `FALSE=0.0`, `UNKNOWN=0.5`.

## Factory constructors

```python
TrinaryCondition.create_true(kind, owner)    # expression = const_true()
TrinaryCondition.create_false(kind, owner)   # expression = const_false()
TrinaryCondition.create_unknown(kind, owner) # expression = const_trinary_unknown()
```

`TransitionKind` enum: `START=1`, `PAUSE=2`, `END=3`, `RESET=4`.

## Invariants enforced on `update_expression()`

1. Expression must be a `Scalar`.
2. All free variables must be `ObservationVariable` — no raw CasADi symbols permitted.
3. A `START` condition may not reference the owner's own `ObservationVariable` (prevents self-starting loops).

## `NodeArtifacts`

The companion dataclass returned by `build()`:

```python
@dataclass
class NodeArtifacts:
    constraints: ConstraintCollection  # QP constraints (default: empty)
    observation: Optional[Scalar]      # observation expression (default: None)
    debug_expressions: List[DebugExpression]  # debugging only
```

If `observation` is `None` and `on_tick()` returns `None`, the node's observation state stays UNKNOWN.

## `ObservationVariable` and `LifeCycleVariable`

Both extend `FloatVariable` from krrood. `ObservationVariable.resolve()` reads `node.observation_state`; `LifeCycleVariable.resolve()` reads `node.life_cycle_state`. These are the only CasADi symbols that may appear in `TrinaryCondition` expressions.

## Related

- **Used by:** [[giskardpy.motion_statechart.graph_node.MotionStatechartNode]], [[concept.motion-statechart]]

## Provenance

- `graph_node.py:64-263` — `TrinaryCondition`, `ObservationVariable`, `LifeCycleVariable`, `NodeArtifacts`, `DebugExpression`.
- `data_types.py:1-54` — `LifeCycleValues`, `ObservationStateValues`, `DefaultWeights`, `TransitionKind`.
