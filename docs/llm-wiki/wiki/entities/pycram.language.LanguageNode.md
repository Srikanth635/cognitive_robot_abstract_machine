---
id: pycram.language.LanguageNode
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/language.py
    lines: [31, 226]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.PlanNode
  - pycram.plans.failures.PlanFailure
  - pycram.fluent.Fluent
  - concept.plan-language
used_by:
  - pycram.plans.factories
  - pycram.plans.failures.PlanFailure
  - concept.plan-language
status: stable
tags: [plan-node, language, combinator, abstract-base]
last_ingest: 2026-05-17
---

_Abstract `PlanNode` subclass that orchestrates children rather than executing robot actions directly. Parent of all plan combinator nodes._

## Purpose

`LanguageNode` is the root of the plan language hierarchy. Its concrete subclasses
implement different **control-flow semantics** over their child plan nodes. They share:

- The `simplify()` override that flattens **same-type** immediate children (using
  exact `type()` equality — not `isinstance`). A `SequentialNode` inside a
  `SequentialNode` is collapsed; a `RepeatNode` inside a `SequentialNode` is not.
- The same `Plan`/`rustworkx` graph as `ActionNode` and `MotionNode` — no separate
  runtime.

## Construction

Do **not** construct these directly. Use the free functions in [[pycram.plans.factories]]
(`sequential`, `parallel`, `try_in_order`, …). Those functions create the node,
build a `Plan`, attach children (using `mount_subplan` for nested language nodes and
`add_child` for leaf nodes), and call `plan.simplify()`.

## Concrete subclass reference

### Sequential branch (`ExecutesSequentially`)

`_perform()` = `[child.perform() for child in self.children]`

| Class | Extra behaviour | Key fields |
|---|---|---|
| `SequentialNode` | First failure propagates immediately. | — |
| `RepeatNode` | Loops the sequential `_perform()` N times. | `repetitions: int = 1` |
| `TryInOrderNode` | Catches `PlanFailure` per child; raises `AllChildrenFailed` if ALL fail. | — |
| `MonitorNode` | Starts a monitor thread in `__post_init__`; can interrupt/pause/resume children. | `condition: Callable\|Fluent`, `behavior: MonitorBehavior` |

### Parallel branch (`ExecutesInParallel`)

`_perform_parallel()` = one `threading.Thread` per child, all joined before proceeding.

| Class | Extra behaviour |
|---|---|
| `ParallelNode` | Runs all in parallel; raises the first failed child's `reason` after join. |
| `TryAllNode` | Runs all in parallel; raises `AllChildrenFailed` only if ALL children fail. |

### Standalone

| Class | Extra behaviour | Key fields |
|---|---|---|
| `CodeNode` | Calls a `Callable` directly. Primarily for debugging/testing. | `code: Callable` |

## MonitorNode details

`MonitorNode.__post_init__` starts a `threading.Thread` that polls `condition.get_value()`
every 0.1 s. The thread is killed when `_perform()` completes (via `kill_event`). If
`condition` is a plain `Callable`, it is wrapped in a [[pycram.fluent.Fluent]] at
construction time.

`behavior: MonitorBehavior` controls what happens when the condition fires:
- `INTERRUPT` (default): calls `self.interrupt()` and stops the monitor.
- `PAUSE`: calls `self.pause()` and stops the monitor.
- `RESUME`: initializes the plan in paused state; calls `self.resume()` when condition
  is met.

## Related

- Concept: [[concept.plan-language]]
- Base class: [[pycram.plans.PlanNode]]
- Failure type: [[pycram.plans.failures.PlanFailure]] (specifically `AllChildrenFailed`)
- Condition wrapper: [[pycram.fluent.Fluent]] (stub)
- Factory entry points: [[pycram.plans.factories]]

## Open questions

- `LanguageNode.simplify()` uses `type(child) != type(self)` — the `!=` is intentional
  (skip if same type, process if different). But the loop body only runs on same-type
  children and skips the rest, so it effectively flattens same-type nesting. Confirm
  that heterogeneous subtrees (e.g. `SequentialNode` containing `TryInOrderNode`)
  remain untouched, as expected.
- `ParallelNode._perform()` raises `child.reason` for the **first** failed child
  (iterates `self.children` in index order). Whether the execution order matters for
  which failure propagates is undocumented.

## Provenance

- `pycram/src/pycram/language.py:31-226` at commit `0528d8cf3`.
