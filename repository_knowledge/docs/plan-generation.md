# Robot simulation plan generation

Prepared: 21 October 2026. Baseline: `cram_wiki` at `baeb7c85646b6d878039838934c8ab5df1868761`.

This development stream builds on the repository knowledge system's phases 0–9. Its phases are named **PG1–PG6** to keep their status separate. The primary output is a complete coraplex simulation program or plan builder, with intentional world-dependent choices expressed through supported grounding mechanisms.

Status: PG1 implemented and locally tested. PG2–PG6 remain pending. Local cache activation is checked against the delivered commit separately. See [delivery progress](progress.md).

## Required result

A request such as “transport the milk to the table” should produce:

- Required robot, scene, simulation mode and asset assumptions, or explicit arguments accepting an existing context.
- Source-supported imports, action parameters and plan construction.
- A complete control-flow structure, including dependencies and supported recovery where needed.
- Explicit semantic constraints and deferred world queries for choices such as objects, grasps and standing positions.
- A simulation entry point and independently defined goal checks.
- Source citations, unresolved inputs, validation results and limits.

Underspecification is intentional late binding, not an unfinished script. Missing task intent must remain an explicit input. A geometric choice may be deferred only where existing coraplex grounding supports it. The generator must not invent identifiers, assets, poses, successful executions or robot capabilities.

The first reference scenario uses the existing PR2 place-setting demonstration under `coraplex/demos/coraplex_bullet_world_demo/`. Its name does not establish physical simulation: its default is kinematic simulation. The requested execution mode must remain explicit. A successful example for one robot does not establish support for another.

## Dependencies and boundaries

| Component | Responsibility | Dependency rule |
| --- | --- | --- |
| Shared records | Task purpose, evidence references, requests and later plan/validation results | Domain records depend on no robotics, model or transport SDK |
| Evidence operations | Select, cite, rank and budget one source snapshot | Depend on shared records and source contracts |
| Repository selection | Identify coraplex's relevant contracts and demonstrations | An outer adapter supplies qualified source identities; it never imports indexed robot code |
| Generation | Convert a task and retrieved evidence into a plan proposal | Use the existing optional model interface and tool loop |
| Grounding | Resolve supported descriptions against the current simulation context | Use existing coraplex/EQL behavior; do not create a second world database |
| Validation | Check construction, run simulation and inspect goal outcomes | An explicit optional runtime boundary owns robotics imports and processes |
| CLI, MCP and notebook | Expose the same evidence and results | Share operations; model requests cannot choose arbitrary executables or cache paths |

No vector database, graph server, separate planner DSL, model router or hardware integration is required for the first version. Introduce an abstraction or dependency only when an implemented use case needs it. New dependencies still require approval. Existing canonical publication pins are not changed by this stream.

## PG1 — Source-grounded planning context

**Purpose:** give a coding agent the correct framework evidence before asking it to produce a full robot plan.

Deliver:

1. A `plan-generation` purpose for the existing `prepare-context` operation and its shared MCP contract.
2. Typed supplementary source references, supplied at composition rather than embedding coraplex names in generic evidence operations.
3. A coraplex reference selection covering action contracts, `Context`, plan execution/composition, deferred grounding, candidate limits and a complete demonstration's setup/build/run methods.
4. Exact snapshot/commit/line/digest citations, shared character budgets, excerpt truncation and explicit missing or ambiguous references.
5. Applicable instruction scope for every additional source path. Profile references remain task-relevance candidates; they do not manufacture resolved call relationships.
6. Notebook tool access to the profile and clear source-only limitations.

Tests must establish source identity, absent/ambiguous reference handling, budgets, instruction scope, preservation of resolved evidence, CLI transport and actual MCP/notebook transport. Run the isolated package suite under the existing memory guard. A real-source pilot checks the configured anchors without importing or running coraplex.

**Exit gate:** an agent can request planning context for an explicit qualified target and distinguish retrieved contracts/examples from omitted evidence. This phase does not select an action from natural language, generate a plan, inspect a live world or claim simulation success.

## PG2 — Action discovery and task selection

Depends on PG1.

- Discover action families from resolved inheritance, including composite actions and their construction methods.
- Retrieve parameter declarations, defaults, inherited contracts, conditions, referenced types and examples. Mark dynamic or unresolved details unknown.
- Distinguish “declared in source,” “has an implementation,” and “supported by this robot/backend.” Static discovery cannot establish the last condition alone.
- Match a task to candidate actions using bounded source queries, keeping alternatives and missing semantic inputs visible.
- Extend the pilot across navigation, pick/place and container operations before broadening to tool actions.

**Exit gate:** independently specified tasks retrieve the expected action contracts and dependencies; unsupported or ambiguous capability claims remain explicit. No action is executed during discovery.

## PG3 — Complete plan proposals

Depends on PG2 and the existing notebook/model interface.

- Introduce a typed request for goal, robot/context selection, scene assumptions, execution mode and allowed underspecification.
- Add a dedicated generation instruction and notebook workflow that use planning evidence before proposing code.
- Generate either a self-contained simulation program or a complete builder accepting a declared existing context. Report which form was requested.
- Preserve supported deferred queries until execution boundaries; do not eagerly resolve choices against a hypothetical future world.
- Return code, source references, required assets/inputs, unresolved bindings and proposed goal checks as a reviewable result.
- Exercise the model/tool interaction with mocked providers; make live provider trials explicitly opt-in.

**Exit gate:** proposals cover all required structural parts for the first scenario and make no execution claim. API and semantic correctness remain subject to PG4.

## PG4 — Construction and simulation validation

Depends on PG3; uses existing coraplex grounding and simulation facilities.

- Validate syntax and referenced API contracts before loading the simulation runtime.
- Construct the plan with the selected robot/world and verify that deferred descriptions use supported mechanisms.
- Run the chosen simulation mode in a bounded process with explicit time, memory and candidate limits.
- Validate the task's independently specified outcome, not merely the absence of an exception. Preserve grounding, planning, collision and goal failures separately.
- Record source and generated-artifact identities, initial scenario, execution mode and final observations.
- Keep robotics imports out of the source service and normal lightweight tests. Runtime tests require the correct configured environment and resources.

**Exit gate:** at least the initial task has a reproducible simulation run and meaningful positive/negative goal checks. Kinematic success and physical-simulation success are separate results.

## PG5 — Bounded repair and useful failure reports

Depends on PG4.

- Feed validated diagnostics and cited contracts back into generation with a bounded repair budget.
- Distinguish a bad API call, an invalid goal assumption, missing grounding candidates and a scene/backend limitation.
- Preserve the original goal checks and failed artifacts; never weaken a test or goal to make generated code appear successful.
- Stop with an actionable result when a required input or capability is absent.

**Exit gate:** controlled failures improve under a measured repair budget or end with a faithful unresolved result. A repair is accepted only after rerunning the relevant validation.

## PG6 — Evaluation and workflow integration

Depends on PG4; evaluate repair separately when PG5 is enabled.

- Compare the same tasks, source revisions, scenes, robots and models with ordinary source navigation and planning context.
- Measure goal success, API validity, citation validity, deferred-binding correctness, tool/model usage, latency and completed-task cost, including index maintenance.
- Test changed APIs, stale caches, absent objects, unreachable goals and unsupported robot/backend combinations.
- Keep generated artifacts, validation receipts and source identities together. Integrate the evidence into coding-agent and notebook workflows without adding raw execution to the read-only MCP tool.
- Reuse the existing release checklist for operating limits, hosted validation and any deliberately selected canonical publication update.

**Exit gate:** useful task success and operating costs are demonstrated on a representative benchmark; regressions and unsupported cases remain visible. No token-saving percentage is assumed.

## Delivery order and first-phase interface

`PG1 -> PG2 -> PG3 -> PG4 -> PG5`, with PG6 evaluating PG4 and optionally PG5.

The first phase extends the current query:

```powershell
./repository_knowledge/.venv/Scripts/python.exe -m repository_knowledge `
  --repository . --database repository_knowledge/.cache/agent-working.sqlite3 `
  --scope working-tree --include-untracked repository_knowledge `
  --include-untracked test/repository_knowledge_test `
  prepare-context coraplex.robot_plans.actions.composite.transporting.TransportAction `
  --profile plan-generation --characters 64000
```

The index must match the selected source and inclusion policy. The result remains a `TaskContext`, with ordinary citations, omissions and authority distinctions. It is evidence for generating a plan, not a generated or validated plan itself.
