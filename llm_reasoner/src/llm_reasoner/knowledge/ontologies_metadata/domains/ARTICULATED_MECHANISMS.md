# Articulated mechanisms — 0.18.0

This seventh reusable domain describes bounded single-coordinate mechanisms and intended configuration/latch actions. The ontology IRI is `http://aicor.knowledge/ontologies/articulated-mechanisms.owl`. New concepts use its `#` namespace and English labels. DUL is inherited through the action core. Direct imports are object manipulation, monitoring requirements and knowledge context; robot-motion structures and phase/condition vocabularies are reused transitively. There is no CoraPlex, SDT, robot, simulator or live-world dependency.

This is a bounded engineering authoring profile, not a universal theory of joints or every opening technique. The open-world OWL vocabulary and strict SHACL/Python profile have different responsibilities. Continuous joints, coupled/multiple coordinates, flexible mechanisms and multiple latches need explicit additional profiles; the current checker rejects them rather than silently dropping requirements.

## Eight synthetic profiles

| Profile | Joint family | Intended completion |
|---|---|---|
| Configuration change | Revolute | Absolute error to the explicit angular target within tolerance |
| Configuration change | Prismatic | Absolute error to the explicit displacement target within tolerance |
| Opening | Revolute | Declared open angular target reached and intended assembly/moving part reported open |
| Opening | Prismatic | Declared open displacement target reached and intended assembly/moving part reported open |
| Closing | Revolute | Declared closed angular target reached and intended assembly/moving part reported closed |
| Closing | Prismatic | Declared closed displacement target reached and intended assembly/moving part reported closed |
| Latch release | Example revolute assembly | Positive released-latch evidence; panel movement is not implied |
| Latch securing | Example revolute assembly | Positive secured-latch and closed-configuration evidence; closure is also an explicit precondition |

The eight records exercise five canonical ActionKind individuals and five concrete description classes. The integrity checker requires the declared action kind to agree with its profile. Revolute/prismatic are joint-kind concepts, not duplicate action classes. A door, drawer or lid can later specialize the descriptive assembly without changing the generic action core.

Closing does not imply latching or locking. Latch release does not imply opening. Securing a latch establishes neither a security rating nor authorization to operate a lock. The closed-configuration requirement for the current securing profile is an explicit engineering choice, not an axiom about every possible latch.

## Mechanism structure and identity

MechanismSpecification refers to an assembly entity specification, distinct fixed and moving part specifications, one JointSpecification and a reference frame. The operator is a separate entity specification. None of these nodes is an observed physical body. JointSpecification identifies its kind, a reused robot-motion CoordinateSpecification, axis, lower/upper limits, open and closed coordinates, and an opening direction of +1 or -1. Open and closed coordinates need not equal the hard limits. Opening need not increase the coordinate.

An AxisSpecification declares the mechanism reference frame, three dimensionless component parameters and an origin using robot-motion PositionTarget. Origin coordinates may be negative. Known axis components must form a unit vector under a squared-norm tolerance of 1e-6; unknown components remain unspecified, and partial components do not establish normalization. A mismatch in frames requires a later explicit transform contract and is rejected here. Axis/origin descriptions do not supply link geometry, calibrated zero configuration, FK/IK or a world transform.

The joint coordinate is angular for a bounded revolute joint and linear for a prismatic joint. Canonical units are RAD and M. Targets/limits may be signed. Known limits must be ordered and nondegenerate; known open/closed/requested targets must lie inside available bounds. The opening-direction convention must agree with known open-minus-closed coordinates. Unknown numeric limits remain valid descriptive parameters but cannot establish physical admissibility.

The error metric names the exact coordinate and target parameter through `errorCoordinate` and `errorTarget`. Error/tolerance/target parameter identities must be distinct. Absolute errors and tolerances must be finite and nonnegative when supplied. A future observer must compute these metrics in the declared coordinate convention; the inspector never calculates them from a robot or mechanism pose.

## Latches and engagement

A handle, when described, must belong to the moving part. A latch must belong to the same mechanism. A configuration action declares `latchReleaseRequired` explicitly. Latch presence alone does not impose release for every technique. The opening/configuration fixtures require release; the closing fixtures deliberately declare a nonblocking latch policy. A true requirement without an identified latch is invalid. Automatic/self-latching or other techniques need appropriately sourced profiles, not a guessed policy.

Where release is required, both the precondition and current actuation-entry/monitor conditions require positive released evidence. Missing Secured evidence is not Released, and PreState release evidence cannot stand in for CurrentState evidence. Closing fixtures do not demand release solely because their assembly has a latch.

ActuationEngaged describes externally assessed effective control of the intended assembly. It is not equivalent to handle contact or a grasp. The domain can link an optional object-manipulation engagement description. Its manipulator must match the operator and its manipulated object must match the handle, or moving part when no handle is specified. This optional reference is not a completed grasp, capability match or executable recipe. No engagement child record is fabricated in the shipped fixtures.

## Conditions, phases and monitoring

All profiles require explicit MechanismReady preconditions and use MechanismFault as an independent CurrentState fault report. Configuration profiles also require PathClear. These predicates are externally assessed requirements, not results from a collision/force solver.

Configuration profiles have Prepare, Engage, Move, Verify, Complete and Abort phases. Initial entry uses the action precondition. Move can be entered only from Engage under CurrentState ActuationEngaged, PathClear and, when required, LatchReleased. Move-to-Verify requires the current completion expression. Verify-to-Complete uses the exact PostState success criterion. Latch actions use Prepare, Operate, Verify, Complete and Abort, with the same distinction between current completion and final verification.

Every nonterminal has an explicit fault path to Abort. The checker rejects bypassed engagement/verification, incorrect phase bindings, wrong guards and missing fault exits. Monitors enumerate the observation dependencies of their conditions and request evidence on Unknown. The Move phase declares an engagement/path/latch invariant and a DuringPhase monitor. This describes a requirement; sampled predicates do not prove continuous preservation.

Phase inspection never selects or executes a transition. Simultaneously supplied completion and fault evidence remains visible as ambiguity. False prerequisites can leave the phase graph unable to begin; unknowns, no-progress, timeouts, sensor coverage, recovery, resource scheduling and termination need later operational contracts. Do not treat this as a complete terminating controller.

## Supplied-state assessment

`tools/articulated_mechanisms.py` provides profile integrity checks and `assess(graph, action, context, facts, quantities)`. Load the ontology import closure together with the selected profile RDF, then validate SHACL and integrity; relation argument signatures are defined by the schema, not duplicated in each example. Explicit context must match the action's KnowledgeContext and scoped description. Inputs must be isolated to that context; there is no sensor acquisition, timestamp qualification, evidence acceptance or automatic world lookup.

Facts use `(scope, relation IRI, ordered entity tuple)` keys. Quantities use `(scope, parameter IRI)` keys with `(value, exact unit IRI)` values. No conversion is inferred. The assessor separately returns precondition, success and fault truth, incompatible open/closed or released/secured reports, and concurrent success/fault reports. A true success expression is not overwritten as false solely because another input conflicts; the conflict is exposed for a later acceptance policy.

`executionReadinessAssessed` and `executed` are always false; `acceptedObservations` is zero. A true result is conditional on supplied inputs, not proof that a real action occurred. Unknown geometry, limits, evidence quality and capability providers are not silently filled.

## Claims, queries and fixtures

`examples/articulated-mechanism-profiles.ttl` contains eight independent synthetic Proposed records. Each reachable descriptive non-type/non-label/non-comment link has an exact included Statement and explicit synthetic evidence. No numeric physical parameter values, actual world binding, observation or accepted assessment is stored. Opening-direction signs and latch policies are labeled fixture conventions, not learned physical facts or ontology defaults. The negative-opening-coordinate and known-axis tests use temporary hypothetical values only.

`tools/build_articulated_profiles.py` rebuilds this example file only; it does not rebuild or overwrite the schema, original inputs, other domains, or registry. Rebuild in a reviewable working copy and compare graphs, since Turtle serialization ordering is not a byte-canonical format.

`queries/articulated-targets.rq` retrieves the selected assembly, parts, joint, frame, coordinate target/unit and latch. `queries/articulated-guards.rq` retrieves that record's phase guards. Callers bind `?record` explicitly. These queries expose descriptions, not accepted measurements, and do not apply evidence/freshness policies.

## Competency questions

| Question | Representation/check |
|---|---|
| Which assembly and moving part are intended? | Explicit distinct assembly/fixed/moving references |
| Is the motion rotational or translational? | Joint-kind/coordinate-kind/unit agreement |
| Does opening mean increasing q? | Explicit opening direction and distinct open/closed coordinate parameters |
| Is the requested coordinate within known bounds? | Conditional finite range checks; Unknown values do not certify limits |
| Does reaching a coordinate alone prove opening? | Opening also requires the intended assembly/part's OpenConfiguration report |
| Does closing imply a secured latch? | No; separate action and evidence |
| Does a latch always need releasing to close? | No; the action declares its release policy |
| Can another handle, coordinate, context or frame substitute? | Binding checks reject it |
| Can movement bypass engagement or verification? | The bounded phase profile rejects that path |
| What if required evidence is missing or incompatible? | Unknown and explicit conflict/ambiguity reports; no execution |

## Reproduce checks

From the package root with requirements installed:

```bash
python tools/validate_articulated_mechanisms.py
python tools/validate_articulated_queries.py
python tools/reason_articulated_mechanisms.py
python tools/validate_tool_material_transfer.py
python tools/validate_imports.py
python tools/review_competency.py
```

Java is required for HermiT. Generic classifications/consistency and optional CRAM/AICOR integration are reported separately. The previously seen SceneNode normalization issue is not repaired by this domain addition. Full AICOR class satisfiability remains unverified. Current release evidence is indexed in `reports/release-0.18.0.json`.

## Next improvements

Populate sourced door/drawer/lid and latch-technique profiles; define calibrated coordinate zero and geometry contracts; add passive resistance, effort/friction/compliance and mechanism-specific failure knowledge through existing physical/causal aspects. Develop explicit coupled/multi-joint and multi-latch profiles, interval-qualified clearance/control evidence, quantitative engagement assessment, guarded recovery and action recipes. Later adapters may bind robot skills, joint-state observers, models and world frames. No working projection, planner, mechanism solver or controller is introduced in this increment.

## Recorded release results

185 domain checks, 26 scoped query cases and 57 import/catalog checks passed. Robot-motion (58), object-manipulation (102), tool/transfer (81) and historical competency (93) checks passed as regressions. Domain and combined generic assembly classification and generic examples/demo consistency passed. The optional AICOR check remains blocked as documented. Of 75 prior RDF files, 74 are byte-identical; only the combined assembly changes.
