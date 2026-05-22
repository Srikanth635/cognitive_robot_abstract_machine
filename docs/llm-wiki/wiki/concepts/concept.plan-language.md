---
id: concept.plan-language
kind: concept
package: pycram
source_paths:
  - path: pycram/src/pycram/language.py
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/plans/factories.py
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.language.LanguageNode
  - pycram.plans.factories
used_by:
  - pycram.language.LanguageNode
  - pycram.plans.factories
status: stable
tags: [plan, language, combinator, sequential, parallel]
last_ingest: 2026-05-17
---

_The plan language is a set of combinator nodes — `SequentialNode`, `ParallelNode`, `RepeatNode`, `MonitorNode`, `TryInOrderNode`, `TryAllNode`, `CodeNode` — that control how a plan graph's children are executed, and a factories module that composes them._

## What it is

The plan language is the **composition layer** of pycram's plan execution model. Rather
than hand-assembling a `rustworkx` graph directly, user code calls combinator functions
from `pycram.plans.factories` (e.g. `sequential([a, b, c])`) to construct a plan
tree whose root is a `LanguageNode` subclass.

Language nodes live in the same `Plan`/`PlanNode` graph as `ActionNode` and `MotionNode`;
they are not a separate runtime. A `LanguageNode`'s `_perform()` orchestrates its
children rather than executing robot behavior directly.

## Two execution modes

| Mixin | Semantics | Concrete classes |
|---|---|---|
| `ExecutesSequentially` | `[child.perform() for child in children]` | `SequentialNode`, `RepeatNode`, `MonitorNode`, `TryInOrderNode` |
| `ExecutesInParallel` | one `threading.Thread` per child, then `join` | `ParallelNode`, `TryAllNode` |

Both groups share the `LanguageNode` ABC as their parent; see [[pycram.language.LanguageNode]].

## The simplify() mechanism

After a plan is assembled, `Plan.simplify()` traverses the graph bottom-up and calls
`simplify()` on each node. `LanguageNode.simplify()` flattens **same-type** immediate
children: if a `SequentialNode` has a `SequentialNode` child, the grandchildren are
reparented directly to the outer node and the redundant middle node is removed.

The check uses `type(child) != type(self)` — **exact type match only**, not
`isinstance`. So a `RepeatNode` inside a `SequentialNode` is NOT flattened.

## API — the factories

All combinators are free functions in `pycram.plans.factories`:

```python
from pycram.plans.factories import sequential, parallel, try_in_order, try_all
from pycram.plans.factories import monitor, repeat, code, execute_single

plan_root = sequential([pick_up_action, navigate_action, place_action])
plan_root = try_in_order([strategy_a, strategy_b, strategy_c])
plan_root = repeat([move_arm], repetitions=5)
```

Each function returns the root `LanguageNode` already attached to a `Plan`. Full
documentation: [[pycram.plans.factories]].

## Failure semantics

| Node | Behavior |
|---|---|
| `SequentialNode` | first failure propagates immediately |
| `ParallelNode` | runs all; raises first failure after all threads join |
| `RepeatNode` | first failure in any iteration propagates |
| `TryInOrderNode` | continues past individual failures; raises `AllChildrenFailed` only if ALL fail |
| `TryAllNode` | runs all in parallel; raises `AllChildrenFailed` only if ALL fail |
| `MonitorNode` | can interrupt/pause/resume the children based on a `Fluent` condition |

`AllChildrenFailed` is a subclass of [[pycram.plans.failures.PlanFailure]].

## Related

- Nodes: [[pycram.language.LanguageNode]]
- Factories: [[pycram.plans.factories]]
- Base graph node: [[pycram.plans.PlanNode]]
- Failures: [[pycram.plans.failures.PlanFailure]]
