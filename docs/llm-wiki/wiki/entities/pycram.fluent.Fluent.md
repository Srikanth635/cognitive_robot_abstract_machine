---
id: pycram.fluent.Fluent
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/fluent.py
    lines: [26, 390]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - pycram.language.LanguageNode
status: stable
tags: [fluent, reactive, condition, monitoring, thread-safe]
last_ingest: 2026-05-17
---

_Thread-safe reactive value wrapper: a live variable that other threads can wait on, combine with logical/arithmetic operators, or subscribe to via callbacks._

## Purpose

Fluents are the reactive primitive for plan-level condition monitoring. `MonitorNode` (a [[pycram.language.LanguageNode]] subtype) polls a `Fluent` every 0.1 s and interrupts the plan tree if the condition fires. Fluents can be chained into networks: `f1.AND(f2)`, `f1 < 5.0`, etc. produce new derived `Fluent` instances that propagate changes automatically.

The key contract: **`get_value()` returns `None` (falsy) or a non-`None` value (truthy)**. `wait_for()` blocks until non-`None`.

## When to use

- **`MonitorNode` conditions** — pass any `Fluent` (or a `Callable` that `MonitorNode` wraps).
- **Cross-thread signalling** — `fluent.set_value(x)` pulses all children and notifies `wait_for` waiters.
- **Complex conditions** — compose with `.AND()`, `.OR()`, `.NOT()`, comparison operators, or `.pulsed()`.
- **Not for:** single-threaded sequential logic — plain Python variables are simpler there.

## Construction

```python
f = Fluent(value=None, name="my_condition")  # starts as None (falsy)
f.set_value(True)   # pulses all children, wakes wait_for()
f.wait_for()        # blocks until non-None
```

## Key methods

| Method | Description |
|--------|-------------|
| `get_value()` | Returns current value (calls it if callable) |
| `set_value(x)` | Mutates value and pulses the tree |
| `wait_for(timeout)` | Blocks until non-`None` |
| `pulse()` | Propagates change notification without changing value |
| `pulsed(handle_missed)` | Returns a child Fluent that becomes `True` on each pulse; `Behavior` controls missed-pulse handling |
| `whenever(cb)` | Registers a callback invoked on every pulse |
| `.AND(other)` / `.OR(other)` / `.NOT()` | Logical composition → new derived `Fluent` |
| Arithmetic / comparison ops | `+`, `-`, `*`, `/`, `<`, `==`, etc. → derived `Fluent` |

## Behavior enum

Controls missed-pulse handling in `pulsed()`:
- `NEVER` — ignore missed pulses.
- `ONCE` — replay body once if pulses were missed.
- `ALWAYS` — replay once per missed pulse.

## Thread-safety

All value access and pulse propagation uses per-instance `Lock` / `Condition`. However, `_children` list is not copy-on-write — structural mutation during iteration could race. This is an existing known risk.

## Related

**Used by:** [[pycram.language.LanguageNode]] (specifically `MonitorNode`)

## Provenance

- `pycram/src/pycram/fluent.py` lines 26–390 (commit `0528d8cf3`) — full `Fluent` class.
- `pycram/src/pycram/fluent.py` lines 12–24 — `Behavior` enum (missed-pulse policy).
