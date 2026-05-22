---
id: pycram
kind: package
package: pycram
source_paths:
  - path: pycram/src/pycram
    commit: 0528d8cf3f93170978915b17e96d311162c5af85
uses: []
used_by:
  - concept.designator
  - pycram.plans.Designator
  - pycram.plans.PlanEntity
  - pycram.robot_plans.ActionDescription
  - pycram.robot_plans.BaseMotion
  - pycram.plans.DesignatorNode
  - pycram.plans.factories.make_node
status: stub
tags: [package, plans, designators, execution]
last_ingest: 2026-05-17
---

_The CRAM-style cognitive robot abstract machine in Python: designators, plans, and plan execution on top of a semantic digital twin and a motion statechart QP backend._

## Purpose

`pycram` is the high-level **planning and execution layer** of this monorepo. It models
a robot's intended behavior as a graph of **plan nodes**, each wrapping a **designator**
(a parametric description of an action or motion). The actual world model and kinematics
come from [[sdt]], and motion execution is handed off to [[giskardpy]] via a Motion
State Chart.

This page is currently a **stub package overview** — it will be filled out as more
ingests cover the package's subareas.

## Top-level subpackages (only those exercised in this ingest)

| Subpackage | Role |
|---|---|
| `pycram.plans` | The plan graph: `Plan`, `PlanNode`, [[pycram.plans.Designator]], [[pycram.plans.DesignatorNode]], [[pycram.plans.factories.make_node]]. |
| `pycram.robot_plans` | Concrete designator hierarchies: [[pycram.robot_plans.ActionDescription]] (actions) and [[pycram.robot_plans.BaseMotion]] (motions). |
| `pycram.designators` | **Empty skeleton** as of this commit (all `.py` files 0 bytes). Likely a future refactor target. See [[concept.designator]] Open questions. |

Other subpackages (`alternative_motion_mappings`, `datastructures`, `external_interfaces`,
`language`, `locations`, `motion_executor`, `ontomatic`, `orm`, `perception`,
`pose_validator`, `querying`, `ros`, `validation`, `view_manager`, `visualization`, …)
are not yet covered. Each will get a touch when its first ingest happens.

## Related

- Concepts: [[concept.designator]]
- Cross-package: this package depends on **sdt** (`World`, `AbstractRobot`, body/pose
  types) and **giskardpy** (`Task` from `motion_statechart.graph_node`). Bridge pages
  will be created when those ingests run.

## Open questions

- The empty `pycram/designators/` directory: deliberate future home or vestige?
- Is there a single public-API entry point (e.g. via `pycram/__init__.py`)? Not yet
  inspected; a later ingest should document it.

## Provenance

- `pycram/src/pycram` directory layout — listed at commit `0528d8cf3`.
