# Transport and carrying — 0.19.0

Update in 0.20.0: ordered pickup–carry–place recipes now compose this carrying window with acquisition and placement; see **ACTION_RECIPES.md**. The original carrying profile and RDF examples are preserved.

The first reusable composition combines **object manipulation** and **mobility/navigation**, with DUL and the generic core imported transitively. Its ontology IRI is `http://aicor.knowledge/ontologies/transport-carrying.owl`; concepts use its `#` namespace and English labels. It imports those two domains and knowledge context, without a robot, world model, CoraPlex, simulator, tool domain or articulated-mechanism dependency.

## Scope and four fixtures

| Fixture | Holding component | Mobility component | Additional requirement |
|---|---|---|---|
| Carrying | Maintained holding | Navigation | Load borne, envelope preserved, load arrives |
| Route carrying | Maintained holding | Route navigation | Same, plus interval route compliance |
| Orientation-constrained carrying | Maintained holding | Navigation | Same, plus interval maximum angular deviation |
| Orientation-constrained route carrying | Maintained holding | Route navigation | Both route and orientation constraints |

Four records exercise two concrete action kinds/classes and an optional OrientationRequirement. TransportDescription is a generic extension point; CarryingDescription is the explicitly grasped/borne technique. RouteCarryingDescription specializes it. The module does not say all transport requires a grasp. Cart transport, pushing, dragging, supported loads, handover and multiple carriers need separate profiles. The current profile begins with a held, borne load. Pickup precedes it and placement/release follow it as separate recipes.

Carrying completion preserves a hold. It neither releases the object nor asserts a stable placement. Carrier arrival alone is insufficient: the **load itself** must be reported in the destination region. The first profile uses the same declared destination region for carrier and load; it does not demand identical positions or an exact placement pose. More specialized carrier-versus-load target contracts can be added later.

## Shared identity and component contexts

The root description links carrier, load, holding manipulator, destination, loaded footprint, HoldingDescription and NavigationDescription (or RouteNavigationDescription). These are descriptive specifications, not actual robots or bodies. The holding object's identity equals the root load; its manipulator equals the root manipulator, which explicitly belongs to the carrier specification. The mobility robot and destination equal the root carrier and destination.

The mobility footprint is the same LoadedFootprintSpecification linked from the parent. It identifies both carrier and load. A plain robot-footprint declaration cannot automatically establish clearance for an extended carried object. LoadedEnvelopeQualified is an explicit TaskContext prerequisite, and LoadEnvelopeMaintained is required over the evaluation window. Geometry, dimensions, meshes, transforms and conservative envelope adequacy remain externally assessed. The module does not calculate a swept volume or certify obstacle clearance.

Holding and mobility descriptions are cloned from the prior synthetic domain fixtures into composition-specific identities. Their original source files are preserved. The parent reuses the children's participant roles, parameter descriptions, capabilities and exact condition-expression nodes rather than independently generating inconsistent copies. The root record snapshots reachable content and exact Proposed statements; there are no independently mutable child stores or accepted observations.

Parent, holding and mobility have three distinct KnowledgeContext IRIs, each scoped to its own action description. Context keys are presentation identifiers, not identity. Input dictionaries remain separate. Conditions shared into the parent express the logical composition; the bounded composition inspector additionally applies each component's scoped interval policy. Flattening raw child facts into one dictionary and directly calling the generic expression evaluator would bypass those policies and is not an accepted transport assessment.

## Intended success and independent failure

Success is the conjunction of:

1. Holding success: endpoint Holds and interval HoldMaintained for the intended manipulator/load.
2. Mobility success: carrier arrival plus position/heading error tolerances; route carrying additionally needs RouteFollowed for its intended route.
3. Composition-local success: the load is in the destination, LoadBorneThroughout and LoadEnvelopeMaintained, plus the optional angular-deviation criterion.

LoadBorneBy is an instantaneous externally assessed carrier/load-support condition; LoadBorneThroughout qualifies the entire carry window. Neither is inferred from contact or endpoint grasp. This distinguishes carrying from merely retaining contact with a separately supported or dragged object. Their truth must be supplied by a qualified observer/model, not invented by the ontology.

The precondition combines each child's preconditions with loaded-envelope qualification and an already borne load. Failure is the disjunction of the two child fault criteria and an explicit LoadTransportFault. A missing success fact is not a fault event; excess orientation deviation can make success false while failure evidence remains unknown. Concurrent success/fault reports are exposed and never arbitrated into execution approval.

## Coextensive evaluation windows

`intervalPolicy` is explicitly `coextensive`. This describes the evidence window for the carry operation, not a scheduler for the children’s full phase graphs. It must not be interpreted as executing holding completely and then beginning navigation. Acquisition/preparation can precede the declared carry window in a later executable recipe.

The inspector requires an explicit root record/context and accepts separate component dictionaries keyed by child description IRIs. Every supplied dictionary must identify its expected child context. An absent child input remains unknown; a supplied unqualified or foreign input is rejected. Parent interval metadata contains context/start/end. Each child must also supply its own interval, with its own context and exactly the same start/end instants. Equivalent timezone offsets compare equal. Timestamps require timezones and start must precede end.

The parent's interval is not silently copied to a child. Children cannot supply a window while the parent window is absent. Shorter holding coverage, different navigation bounds or a foreign context are errors. Without all three bound windows, aggregate root load-support/envelope facts and maximum-deviation metrics are withheld. The existing holding inspector also withholds unqualified HoldMaintained; route navigation withholds unqualified RouteFollowed. Endpoint facts alone therefore cannot establish carrying success.

False endpoint evidence can still establish an unmet criterion when interval evidence is missing. Interval metadata scopes the question; it does not prove continuous observation, sampling coverage, correctness, calibration or freshness. A synthetic or LLM-proposed fact is not accepted sensor evidence through this API. AcceptedObservations remains zero.

## Orientation constraints

OrientationRequirement links the same load, an explicit reference frame, a robot-motion quaternion XYZW OrientationTarget and a maximum-deviation/tolerance parameter pair in radians. It constrains orientation only: a full pose-fixed-to-world requirement would incorrectly forbid translation during transport. No such fixation requirement is imposed.

The deviation is the externally assessed **maximum angular deviation over the carry window**, not endpoint error, mean error or a computed pose difference. Missing aggregate evidence remains unknown even when an endpoint orientation metric is supplied. A later adapter must establish that the metric uses the intended target, frame, time window and angular-distance convention. No transformation or quaternion-error calculation is implemented here.

Target components are dimensionless and may remain unspecified. Known complete quaternions must be normalized within a squared-norm tolerance of 1e-6. Known deviation/tolerance values must be finite and nonnegative with exact RAD unit identifiers. No numerical orientation or tolerance is inserted in the fixtures. This first profile constrains full orientation about a quaternion target; tilt-only upright constraints with free yaw, gravity-axis constraints and time-varying orientation targets require explicit extensions.

Conditional success from a supplied deviation metric does not establish that an incomplete target has been calibrated or bound correctly. Execution feasibility and readiness are never certified.

## Monitoring and phase boundary

A composition-scoped monitor attaches to the cloned mobility execution phase. It checks CurrentState Holds, LoadBorneBy and LoadEnvelopeClear for the same participants and loaded footprint. It explicitly names the parent context. False or Unknown declares RequestSuspension. The monitor does not stop a controller or issue a robot command.

The child phase graphs remain inspectable and retain their own fault/monitor requirements. This increment adds no global scheduling, resource locks, synchronization controller, automatic handoff, timeout policy, retry executor or operational placement phase. Continuous control and collision guarantees require later evidence/actuation adapters. The guard preview is never execution permission.

## Reference API

Load the import closure with the examples, run SHACL and `carrying_errors`, then use:

```python
assess(graph, record_iri, parent_context_iri,
       components={
           holding_description_iri: {
               "context": "<holding context>",
               "interval": {"context": "<holding context>",
                            "start": "<timezone-bearing start>",
                            "end": "<timezone-bearing end>"},
               "facts": {...},
           },
           mobility_description_iri: {
               "context": "<mobility context>",
               "interval": {"context": "<mobility context>",
                            "start": "<same start instant>",
                            "end": "<same end instant>"},
               "facts": {...}, "quantities": {...},
           },
       },
       interval={"context": "<parent context>",
                 "start": "<same start instant>",
                 "end": "<same end instant>"},
       facts={...}, quantities={...})
```

Keys for facts are `(scope, relation IRI, ordered entity tuple)`. Quantity keys are `(scope, parameter IRI)` with `(value, exact unit IRI)` values. Root facts/quantities are scoped separately from each child dictionary. No unit conversion, source lookup or implicit latest-record selection occurs. Outputs include each child's assessment, local success, parent precondition/success/failure, coextensive-window binding and simultaneous-success/fault flags. Executed, executionReadinessAssessed and releasePerformed are false; placementAssessed is false.

## Provenance, queries and reproduction

All four fixtures remain Proposed with explicit synthetic evidence. Every reachable substantive non-type/non-label/non-comment link has an exact included Statement. There are no physical parameter values, actual bindings or source-validated claims. Historical domain and pouring records are unchanged. The example builder regenerates only `examples/transport-carrying-profiles.ttl` from the preserved domain fixtures; it does not alter schemas or the original fixtures.

`queries/transport-components.rq` exposes parent/child identities and distinct contexts. `queries/transport-orientation.rq` returns orientation requirements only where declared. Bind `?record` explicitly. SPARQL retrieval does not apply interval/evidence policy.

From the package root:

```bash
python tools/validate_transport_carrying.py
python tools/validate_transport_queries.py
python tools/reason_transport_carrying.py
python tools/validate_object_manipulation.py
python tools/validate_mobility_navigation.py
python tools/validate_imports.py
python tools/review_competency.py
```

Java is required for HermiT. The combined consistency check uses a bounded 6144 MB heap and 180-second subprocess timeout after the initial 2048 MB run exhausted its heap; that initial result is preserved separately. Current reasoning status is indexed in `reports/release-0.19.0.json`. A resource-limited check is not a logical inconsistency or a passed check. The optional AICOR normalization limitation remains separately recorded, and full AICOR class satisfiability is unverified.

## Next refinements

Populate sourced load/hold/envelope knowledge; define geometry and observation-coverage contracts; bind capabilities and frames; add supported/cart, multiple-grasp and handover techniques explicitly. Compose pickup and placement around this carrying window in later recipes. Add source-backed acceleration, load dynamics, stability, compliance and contents-sensitive constraints through existing physics/control/causal aspects. No new working projection, execution adapter or transport planner is introduced here.

## Recorded release results

159 composition checks, 14 scoped query cases and 59 import/catalog checks passed. Object-manipulation (102), mobility/navigation (104) and historical competency (93) regressions passed. The composition and combined generic assembly classify with no unsatisfiable named classes. Combined generic example/demo consistency passed with a 6144 MB heap; the initial 2048 MB heap failure remains recorded. Optional AICOR integration remains blocked by the previously documented normalization issue. Of 78 prior RDF files, 77 are byte-identical; only the combined assembly changes.
