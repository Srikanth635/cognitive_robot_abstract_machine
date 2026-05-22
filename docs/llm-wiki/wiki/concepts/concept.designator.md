---
id: concept.designator
kind: concept
package: pycram
source_paths:
  - path: pycram/src/pycram/plans/designator.py
    lines: [19, 82]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/actions/base.py
    lines: [41, 137]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
  - path: pycram/src/pycram/robot_plans/motions/base.py
    lines: [22, 63]
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses:
  - pycram
used_by:
  - pycram.plans.Designator
  - pycram.plans.Plan
  - pycram.plans.PlanNode
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.BaseMotion
  - pycram.plans.DesignatorNode
  - pycram.plans.factories.make_node
status: stable
tags: [designator, plan, action, motion, core-concept]
last_ingest: 2026-05-17
---

_A **designator** is a parametric description of something the robot can do — an action
or a motion — that is managed by a plan node and executed in the context of a plan,
robot, and world._

## Purpose

The designator pattern separates **what** the robot is asked to do (the parameters and
high-level intent) from **how** it is executed (the plan-graph machinery and the motion
backend). A `Designator` is a plain dataclass; the corresponding [[pycram.plans.DesignatorNode]]
in the plan graph is what actually performs it. This keeps the parametric description
serializable and inspectable while the execution context is mutable and graph-shaped.

In this codebase, "designator" is a roof concept with two practical subtrees:

- **Actions** ([[pycram.robot_plans.ActionDescription]]) — *builders for plans*. An
  action can produce a subplan of motions and other actions via its `execute()` method.
  Actions carry pre/post conditions expressed as symbolic expressions (krrood).
- **Motions** ([[pycram.robot_plans.BaseMotion]]) — *builders for Motion State Charts*.
  A motion creates exactly one goal (a [[giskardpy]] `Task`) and is leaf-level: motions
  do not create other motions or actions.

## When to use

- **Define a new ActionDescription subclass** when introducing a new high-level
  behavior (e.g. `PickUp`, `Transport`) that decomposes into motions or other actions.
- **Define a new BaseMotion subclass** when introducing a new primitive command for a
  robot (e.g. open gripper, move TCP) that maps directly to a giskardpy goal.
- **Do not** subclass `Designator` directly outside these two trees — the plan-graph
  dispatch in [[pycram.plans.factories.make_node]] only recognises the two leaf trees.

## How it fits together (data flow)

```
ActionDescription / BaseMotion       <-- the parametric "what"
        │ (wrapped by)
        ▼
ActionNode / MotionNode              <-- plan-graph node ("DesignatorNode")
        │ (lives in)
        ▼
Plan                                 <-- the executable graph
        │ (motions collected into)
        ▼
MotionExecutor → giskardpy Motion State Chart
```

The wrapping is done by [[pycram.plans.factories.make_node]], which dispatches on type.

## Key invariants

- A `Designator` does not know its own plan, robot, or world until it is wrapped. Its
  `plan` / `robot` / `world` / `context` properties are **delegated through
  `plan_node`**; accessing them on an unattached designator raises `ContextIsUnavailable`.
- The `plan_node` back-reference is set by `DesignatorNode.__post_init__`, not by the
  Designator itself.
- Motions never spawn motions or actions. Actions may spawn either.

## Related

- Subclasses: [[pycram.robot_plans.ActionDescription]], [[pycram.robot_plans.BaseMotion]]
- Wrappers: [[pycram.plans.DesignatorNode]] (parent of `ActionNode`, `MotionNode`)
- Construction: [[pycram.plans.factories.make_node]]
- Package: [[pycram]]
- Cross-package (future bridge pages): `sdt.world.World`, `sdt.robots.AbstractRobot`,
  `giskardpy.motion_statechart.graph_node.Task`, `krrood.entity_query_language.*`

## Open questions

- `pycram/src/pycram/designators/` is an empty skeleton (`designator.py`,
  `location_designator.py`, `specialized_designators/probabilistic/probabilistic_action.py`
  all 0 bytes at commit `0528d8cf3`). Is this a planned future home for the Designator
  hierarchy (i.e. a refactor target), or vestigial scaffolding? The "real" base class
  lives in `pycram.plans.designator`, not here. Worth confirming with the team before
  the next ingest that touches this area.
- Are there other `Designator` subclasses outside the `actions` / `motions` trees that
  this ingest missed? grep at commit `0528d8cf3` found 5 references to `class .*Designator`
  but only two new subclasses; the others were `ActionNode`/`MotionNode` references
  and an ORM mapping. Worth re-checking when ingesting `pycram.orm`.

## Provenance

- `pycram/src/pycram/plans/designator.py:19-82` — the abstract base.
- `pycram/src/pycram/robot_plans/actions/base.py:41-137` — `ActionDescription`.
- `pycram/src/pycram/robot_plans/motions/base.py:22-63` — `BaseMotion`.
