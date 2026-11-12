> Import architecture updated in release 0.6.0: see `MODULAR_IMPORTS.md` at the package root. Earlier import-chain/open-file instructions below are superseded. Vocabulary descriptions still apply.

# Phases, transitions and monitoring — release 0.3.0

Two modules extend the shared action representation. They reuse the existing PhaseSpecification, ConditionSpecification, OutcomeSpecification, CapabilityRequirement, entity specifications, claims and evidence; they do not define a competing action representation.

| Module | Main responsibility |
|---|---|
| phases-transitions.ttl (`pt:`) | Phase membership, initial phase, guarded transitions, terminal dispositions, entry/exit requirements, invariants and expected outcomes |
| monitoring-requirements.ttl (`mr:`) | What condition to check, when to check it, which observations it requires and declared responses to true/false/unknown |

Both use HTTP IRIs under `http://aicor.knowledge/ontologies/`, with `.owl#` namespaces and English labels. They are not hosted at these addresses. The XML catalog resolves local imports.

Import chain: monitoring-requirements → phases-transitions → conditions-outcomes → participants-relations → action-description-core → DUL. Open `observation/monitoring-requirements.ttl` to load the full generic stack. AICOR remains an optional external integration dependency through the existing bridge.

## Phase structure

An ActionDescription links to a PhaseStructure, which owns a finite graph of PhaseSpecifications and PhaseTransitions. Each transition has one source, one target and one structured guard condition. Ordinary, failure and recovery transitions are distinct categories. There is one initial phase in the validation profile, and at least one terminal phase must be graph-reachable from it.

Phases reuse the core's PhaseSpecification class. TerminalPhase, SuccessfulTerminalPhase, FailedTerminalPhase and RecoveryPhase specialize it. Success/failure terminal labels specify intended dispositions; they do not assert that an action has succeeded or failed.

A phase can reference:

- Entry condition: a declared requirement for entry.
- Exit condition: a declared requirement for normal phase completion.
- Invariant conditions: intended conditions during the phase.
- Expected outcomes: the existing before/after outcome descriptions.
- Monitoring requirements: checks associated with this phase.

These declarations are not an operational execution policy. The reference guard assessor evaluates only guards. In particular, it does not silently combine guards with entry, exit or invariant conditions. A future execution adapter must specify how those requirements interact with failure/recovery transitions. Initial precheck phases and exceptional exits need explicit operational policies before execution.

## Branching, recovery and topology

The representation supports branching and cycles. Recovery does not require flattening a phase graph into a single ordered list. The checker rejects missing endpoints, endpoints outside the owning structure, multiple owners for a transition, unreachable phases, outgoing edges from terminal phases and nonterminal dead ends.

Graph reachability does not prove that guards can be satisfied. A recovery cycle does not imply bounded retries or guaranteed termination. Retry budgets, timeouts, scheduling, concurrency, hierarchical phase execution and phase-duration models are not implemented in this release.

The guard preview reports:

| Assessment | Meaning |
|---|---|
| terminal | Declared terminal phase, with no outgoing transitions |
| no_guard_true | All outgoing guards explicitly false |
| incomplete | At least one outgoing guard unknown, with at most one known true |
| ambiguous | More than one outgoing guard known true |
| one_guard_true | One known true guard and no unknown alternatives |

It always returns `selected_transition: null`. Even one known true guard is not permission or proof of readiness to execute a transition. There is no hidden priority rule between ordinary, failure and recovery branches.

## Monitoring requirements

Each MonitoringRequirement names a condition, a monitoring point, its observation dependencies and explicit response categories for True, False and Unknown. MonitoringPoint values are PhaseEntry, DuringPhase, PhaseExit and TransitionCheck.

ObservationRequirement has two concrete forms:

- RelationObservation: a relation pattern and its evaluation scope.
- QuantityObservation: a parameter specification and its evaluation scope.

An observation can require an abstract capability such as quantity estimation. It names no camera, force sensor, simulator, robot or API. No measurement frequency, latency or precision is invented.

The dependency checker walks the condition expression and requires every distinct relation/quantity leaf exactly once. Constant expressions need no external observations. Missing, duplicate, foreign or wrong-scope dependencies are rejected. Shared ObservationRequirement nodes can be reused across monitors.

Monitoring responses are declarations, not commands. The reference assessor returns the category but never executes it. Response categories include reporting satisfaction, reporting an unsatisfied condition or violation, requesting evidence, requesting suspension/recovery, and reporting unknown. Unknown must use RequestEvidence, RequestSuspension or ReportUnknown in the current validation profile; it cannot become ReportSatisfied.

Examples distinguish a false readiness condition (not yet ready) from unknown readiness. Sampling a condition during a phase cannot prove that an invariant held continuously. Freshness, sensor uncertainty, evidence acceptance, anomaly detection, latching and temporal event detection remain adapter/research concerns.

## Scope and identity

Participant and parameter identifiers are reused from release 0.2.0. New CurrentState expressions are explicit; existing PreState and PostState expressions are not silently reinterpreted. The adapter supplying evaluation inputs must establish which action context and snapshot each scope denotes. The reference evaluator consumes one supplied context at a time and provides no multi-episode store.

The examples introduce `pourRecordV3` and `graspRecordV3`, each with a new ActionDescription. Previous files and records are unchanged. The new records retain relevant earlier condition claims and add Proposed claims for the phase structure, guards and monitor conditions. All evidence remains explicitly synthetic. Annotation records identify the new contribution; the earlier record provides the history for reused claims.

Direct template links describe the authored proposed structure. They do not assert event occurrences. Claim coverage demonstrates provenance for the structure's initial phase, transition guards and monitor conditions; it is not yet a statement-complete extraction/provenance pipeline for every structural triple.

## Examples

**Pouring** has Prepare, Transfer, Verify, Complete and Abort phases. Explicitly false preconditions can lead to Abort; unknown preconditions produce no selected path. Transfer-to-Verify requires sufficient delivered amount and no excessive spillage, both using CurrentState inputs. Excessive spill has a separate failure guard. Verify checks the existing PostState success expression.

Four monitoring requirements cover transfer spillage, readiness, the transition check and final verification. Missing evidence requests more evidence. The source is not required to become empty. Numeric values occur only in evaluator test fixtures, not as recommended pouring parameters.

**Grasping** has Prepare, Establish, Verify, Recover, Complete and Abort phases. Control plus contact in CurrentState makes advancement to Verify a candidate. Explicit failure of PostState verification leads to Recover. Recovery checks current graspability and may describe a retry or failure disposition. This cycle is intentionally a conceptual example; no autonomous retry policy is supplied.

Three monitoring requirements cover establishing a hold, verification and recovery. The examples provide no timeout for a phase whose completion condition stays false; they must not be treated as complete executable plans.

## Files and run commands

| File | Purpose |
|---|---|
| aspects/validation/temporal-shapes.ttl | SHACL constraints for phase and monitoring structure |
| tools/temporal_runtime.py | Topology/dependency checks and non-executing guard/monitor assessment |
| examples/phase-actions.ttl | New conceptual action descriptions and v3 knowledge records |
| examples/phase-projections.json | Readable phase, guard and monitor views with claim references |
| queries/phase-monitoring.rq | Query which evidence inputs each action's monitors need |
| reports/temporal-* | Validation, reasoning and inventory evidence |

From the package root with requirements installed:

```bash
python tools/validate_temporal.py
python tools/project_temporal.py
python tools/reason_temporal.py
python tools/check_temporal_integration.py
```

HermiT requires Java. Validation combines the core, previous aspect shapes, new shapes, finite-expression checks, phase-topology checks and monitoring-dependency checks. HermiT classification covers all four aspects plus core and DUL; the larger AICOR integration check covers ontology consistency only.

## Next development boundary

The next substantial aspect can be causal/mechanistic knowledge: connect phases and conditions to proposed effects, mechanisms, assumptions and supporting evidence. Physics/simulation model specifications should then refer to those same parameters and outcomes. Neither should treat a generated causal narrative or a simulation run as a universal physical guarantee.

Detailed Minsky-frame/image-schema projections, EQL integration, LLM annotation, actual SDT bindings and executable CoraPlex adapters remain future work.

## Recorded validation result

- All 34 new validation and semantic checks passed, including malformed graphs, incomplete observation dependencies, unknown truth handling and ambiguous guards.
- Full HermiT classification of all four aspect modules with core and DUL passed, with no unsatisfiable named classes.
- Combined ontology consistency with AICOR, the bridge and all examples passed; full combined classification remains unverified.
- Projections contain 2 records, 11 phases, 13 transitions and 7 monitors. The observation query returns 19 rows.
- The maintained 0.2.0 ontology files, shapes and earlier example data were verified byte-for-byte unchanged.
