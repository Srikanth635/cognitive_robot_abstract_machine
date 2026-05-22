---
id: pycram.robot_plans.BaseMotion
kind: entity
package: pycram
source_paths:
  - path: pycram/src/pycram/robot_plans/motions/base.py
    lines: [22, 63]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram.plans.Designator
  - concept.designator
  - giskardpy.motion_statechart.graph_node.Task
used_by:
  - bridge.pycram-giskardpy
  - pycram.plans.factories.make_node
  - pycram.plans.MotionNode
  - pycram.robot_plans.motions.gripper
  - pycram.alternative_motion_mapping.AlternativeMotion
  - pycram.robot_plans.motions.container
  - pycram.robot_plans.motions.misc
  - pycram.robot_plans.motions.navigation
  - pycram.robot_plans.motions.robot_body
status: stable
tags: [designator, motion, abstract-base, motion-statechart, giskardpy]
last_ingest: 2026-05-18
---

_Abstract subclass of [[pycram.plans.Designator]] for motions: a "builder for Motion State Charts" that creates exactly one goal._

## Purpose

`BaseMotion` is the abstract base for **all motions** in pycram. A motion is leaf-level
in the [[concept.designator]] hierarchy: it does **not** create other motions or
actions, and it produces exactly one [[giskardpy]] goal (a `Task` from
`giskardpy.motion_statechart.graph_node`). Motions are collected by their parent
`ActionNode` and merged into a single Motion State Chart for execution.

It also supports **alternative mappings**: a motion can be transparently swapped for an
`AlternativeMotion` based on the current robot via
`AlternativeMotion.check_for_alternative`. The alternative is constructed with the same
parameters and inherits the original's `plan_node`.

## When to use

- **Subclass** when adding a new robot primitive that maps to a single QP goal (e.g.
  `MoveGripperMotion`, `MoveToolCenterPointMotion`). Subclasses must implement:
  - `perform()` — what happens at execution time (delegates to the process module).
  - `_motion_chart` (property) — returns the `Task` for this motion.
- **Do not** spawn other motions or actions from inside a motion. That violates the
  leaf invariant.

## Construction

```python
@dataclass
class MyMotion(BaseMotion):
    target: Pose
    def perform(self):
        ...
    @property
    def _motion_chart(self) -> Task:
        return SomeGiskardTask(...)
```

Concrete examples in `pycram/src/pycram/robot_plans/motions/`:
`MoveGripperMotion`, `MoveToolCenterPointMotion` (gripper.py), and motions in
`container.py`, `navigation.py`, `robot_body.py`, `misc.py`. Entity pages for these
will appear in later ingests.

## Key methods and attributes

| Name | Kind | Notes |
|---|---|---|
| `perform()` | abstract method | Overwritten by each motion; sends this designator to the process module for execution. |
| `motion_chart` | property → `Task` | Returns the alternative motion's chart if one is registered for the current robot; otherwise `self._motion_chart`. Constructs the alternative with the same parameters and binds its `plan_node`. |
| `_motion_chart` | abstract property → `Task` | Subclass hook. The giskardpy `Task` representing this motion's single goal. |
| `get_alternative_motion()` | method → `Optional[Type[AlternativeMotion]]` | Lookup via `AlternativeMotion.check_for_alternative(self.robot, self.__class__)`. |

Module-level: `MotionType = TypeVar("MotionType", bound=BaseMotion)`.

Type-var on the module: `T = TypeVar("T", bound=AbstractRobot)` (declared but not used
in the visible body — kept here as it appears in the source).

## Related

- Parent: [[pycram.plans.Designator]]
- Sibling: [[pycram.robot_plans.ActionDescription]]
- Wrapper: `MotionNode` (subclass of [[pycram.plans.DesignatorNode]])
- Concept: [[concept.designator]]
- Dispatcher: [[pycram.plans.factories.make_node]]
- External: `giskardpy.motion_statechart.graph_node.Task`,
  `semantic_digital_twin.robots.abstract_robot.AbstractRobot`,
  `pycram.alternative_motion_mapping.AlternativeMotion`

## Open questions

- The alternative-mapping flow uses `inspect.signature(self.__init__).parameters` to
  copy parameters. This will misbehave if a dataclass uses default factories or
  init-only fields. Worth a focused test in a future ingest.

## Provenance

- `pycram/src/pycram/robot_plans/motions/base.py:22-63` at commit `0528d8cf3` — full
  class definition.
- Line 65: `MotionType` alias.
