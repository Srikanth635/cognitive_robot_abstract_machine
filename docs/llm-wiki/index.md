# Wiki index

Catalog of every wiki page. One line per page: `[[id]] — kind — one-line summary`.
Newly created or updated pages are added/edited on every ingest (see [CLAUDE.md §6](CLAUDE.md)).

Entity listings have been split into per-package index files for navigability (schema v3):
- **pycram entities** → [[index-pycram]]
- **SDT entities** → [[index-sdt]]
- **giskardpy entities** → [[index-giskardpy]]

---

## Packages

- [[pycram]] — package — Cognitive robot abstract machine: plans, designators, plan execution.
- [[sdt]] — package — Semantic Digital Twin: kinematic world tree, robot models, semantic annotations, FK/IK.
- [[giskardpy]] — package — Reactive motion controller: Motion State Chart orchestration + QP-based whole-body control.

## Concepts

- [[concept.designator]] — concept — Parametric description of an action/motion that a plan node manages and executes.
- [[concept.plan-language]] — concept — Combinator nodes (sequential, parallel, monitor, try_*) that compose plan trees.
- [[concept.world]] — concept — SDT kinematic world: tree of Body/Region nodes joined by Connection edges; Pose as symbolic CasADi HTM.
- [[concept.motion-statechart]] — concept — MSC directed graph: compiled trinary lifecycle + observation states drive QP constraint selection each control cycle.
- [[concept.krrood-eql]] — concept — krrood Entity Query Language: symbolic variables, selective/generative queries, pre/post-conditions, underspecified grounding.
- [[concept.forward-kinematics]] — concept — SDT symbolic FK: BFS expression chain + batched CasADi compilation; WorldState as zero-copy state bus for both FK and QP.
- [[concept.semantic-annotation]] — concept — SemanticAnnotation overlay on kinematic entities; mixin traits; RDR-based inference via WorldReasoner; EQL predicate integration.
- [[concept.qp-controller]] — concept — QP pipeline: constraint collection → MPC formulation → QPSolver → joint velocity commands each tick.

## Bridges

- [[bridge.pycram-sdt]] — bridge — pycram ↔ SDT: Context injection, world mutation in pick/place, SDT types in failure dataclasses.
- [[bridge.pycram-giskardpy]] — bridge — pycram ↔ giskardpy: motion designators produce Tasks; ActionNode executes the MSC via MotionExecutor.
- [[bridge.sdt-giskardpy]] — bridge — giskardpy ↔ SDT: World as execution substrate, DegreeOfFreedom as QP variables, FK binding, SDT spatial types as constraint language, SDT collision infrastructure.
