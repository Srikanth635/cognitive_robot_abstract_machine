# Object manipulation — 0.14.0

> Updated in 0.22.0: see [PRIMARY_ACTION_GENERATION_GUIDE.md](PRIMARY_ACTION_GENERATION_GUIDE.md) and `reports/release-0.22.0.json`. The dated development history and counts below retain their original scope.

The second reusable domain is `object-manipulation.ttl`:

`http://aicor.knowledge/ontologies/object-manipulation.owl`

It imports robot motion, phase/transition descriptions, monitoring requirements and knowledge context. DUL is the foundation through the shared core. No CoraPlex, SDT, ROS, particular robot or world instance is imported. The combined `all-action-knowledge.ttl` assembly now includes both reusable domains; the generic aspect assembly stays unchanged.

## Seven action families

| Family | Required success state in the bounded authoring profile |
|---|---|
| Reaching | Manipulator at the intended target region |
| Grasp acquisition | An assessed hold on the intended object and manipulator/object contact |
| Holding | Hold maintained throughout the evaluation interval and present at the endpoint |
| Pickup | Hold present, original support no longer supports the object, and object clear of that support |
| Placement | Object stably supported by destination support, at the intended target region, no longer held, and manipulator clear |
| Release | Manipulator no longer holds the object |
| Fixation | Holding maintained and object pose maintained relative to an anchor frame throughout the interval |

These are explicit engineering profiles of intended outcomes, not universal axioms saying every manipulation must use one technique. The open-world ontology supports extension; the strict shapes and Python checker validate these seven current profiles. Placement is the supported-place-and-clear variant. Release has no implicit support, clearance or destination guarantee. Grasp acquisition does not require removal from support. A closed gripper or contact alone does not establish a hold.

## Reused structures

- `ObjectManipulationDescription` specializes `ac:ActionDescription` (DUL Description).
- Action kinds are individuals of `ac:ActionKind` (DUL Task), matching the existing core convention.
- Object, manipulator, source/destination support, target region and frame are entity specifications. They are not observed physical objects.
- Core role specifications identify participants. New target regions can refer to robot-motion pose targets and explicit frame specifications.
- `ManipulationMotionRequirement` is an action component referring to a canonical robot-motion action kind, controlled entity and optional target region. It is an abstract requirement, not a robot-motion execution or controller instance.
- Existing `pr:Accessible`, `pr:Graspable`, `pr:InContactWith` and `pr:Supports` are reused. More specific domain relation definitions include Holds, HoldMaintained, ClearOfSupport and PoseMaintained. `pr:Controls` has broader semantics; it is not silently equated with the new explicit hold predicate.
- Existing condition expressions encode conjunction and negation. A missing Holds fact remains Unknown; it is never converted into NotHeld.

Motion requirements deliberately stop before trajectory planning. The example grasp/release profiles request gripper-configuration motion; alternative techniques can supply different requirements. Pickup's acquisition phase does not itself implement a complete grasp plan. Relating target object pose to manipulator pose still needs grasp geometry and frame transforms supplied by a later adapter.

## Conditions and phases

Examples include formal preconditions, goals, success criteria, explicit fault criteria, branching phase descriptions and observation requirements. Preconditions vary by profile: accessibility/graspability, prior support for pickup or prior holding for release/placement/holding. They are partial domain prerequisites; actuator availability, detailed collision constraints and physical feasibility remain additional checks.

Most profiles have Prepare, Execute, Complete and Abort phases. A precondition guards entry; success and explicit fault conditions guard completion and abort. The pickup profile inserts **Acquire** before lifting. An assessed current hold guards entry into lifting; an explicitly failed grasp can branch to Abort. Hold monitoring attaches only to the lifting phase, avoiding a false hold-loss report before acquisition.

Hold/fixation monitoring uses current-state hold evidence and requests suspension when a required hold is absent. Fault monitors use explicit fault predicates such as GraspFailed, HoldLost or SupportUnstable. Failure is not defined as the mere absence of success evidence. Unknown evidence requests more evidence; no monitor response is executed by these tools.

A phase graph is an inspectable specification. It does not establish guard satisfiability, termination, scheduling, timeout policy or real execution. False preconditions can leave a specification unable to begin. Simultaneously supplied success/fault facts can enable incompatible branches; the existing phase inspector reports ambiguity and never chooses a transition.

## Interval and context contract

Holding and fixation have explicit `evaluationContext` references to `kc:KnowledgeContext`. The context describes the selected action description. Their interval predicates denote externally assessed properties over an entire supplied interval, not snapshots.

`assess` requires caller-supplied interval metadata for these predicates:

```python
interval = {
    "context": "<the selected profile's context IRI>",
    "start": "2026-11-11T10:00:00Z",
    "end": "2026-11-11T10:01:00Z",
}
```

Times must carry timezones and start must precede end. The context IRI must match the action's context. No actual interval is asserted in the example RDF. Without supplied interval metadata, aggregate HoldMaintained/PoseMaintained facts are withheld from evaluation, so a positive endpoint observation alone cannot establish sustained success.

This bounds the question being evaluated. It does not prove continuous monitoring, measurement coverage or timestamp alignment. The caller must qualify observations to the same context, interval, frames and thresholds. Freshness, provenance acceptance and evidential adequacy remain separate policies; synthetic or LLM-produced assertions must not silently become accepted sensor evidence.

## Fixtures, claims and reference inspection

`examples/object-manipulation-profiles.ttl` contains seven hypothetical records. All new claims are Proposed and evidence explicitly identifies synthetic examples. Every reachable descriptive non-type/non-label link is represented by an exact statement included in the corresponding record. No original record, DUL or AICOR source is rewritten.

`tools/object_manipulation.py` provides:

1. A bounded integrity checker for participant identities, role bindings, required positive/negative success atoms, state scopes and motion target/control bindings.
2. A reference assessment of precondition, success criterion and fault criterion from caller-supplied boolean facts.
3. Interval-context checks and an explicit flag when success and fault are both reported.

Validate shapes and integrity before assessment. The supported expression profile is conjunctions of atomic predicates and negated atoms. This is not a general logical entailment checker. Required predicates must bind the intended entities and the correct state scopes.

Facts use the existing evaluator key `(scope, relation definition, ordered entity-specification tuple)`. PreState, CurrentState and PostState remain distinct. No world lookup, sensor acquisition, friction/grasp model, collision solver, pose computation or execution is performed. `success` means that the supplied success expression evaluates True; preconditions and faults are returned separately. It never certifies a completed real action or execution readiness. The `successAndFaultReported` flag identifies concurrent reports needing investigation, not a proved logical inconsistency.

## Competency checks

The test suite covers:

- Unknown evidence for every profile; satisfaction and violation of required states.
- Gripper closure/contact versus grasp, hold versus pickup, stable placement versus release.
- Wrong objects/supports/targets, wrong motion bindings and mismatched state scopes.
- Interval context, missing interval bounds, zero-duration intervals and endpoint-only evidence.
- Pickup acquisition/hold-monitor ordering and ambiguous acquisition evidence.
- Exact statement coverage, Proposed statuses and absence of world/framework dependencies.

Run with requirements installed from the package root:

```bash
python tools/validate_object_manipulation.py
python tools/reason_object_manipulation.py
python tools/validate_robot_motion.py
python tools/validate_imports.py
python tools/review_competency.py
```

Java is required for classification. The new reasoning suite classifies the standalone domain and combined assembly, checks generic examples consistency, and records optional AICOR integration separately. The AICOR normalization issue documented in release 0.13 remains an explicit integration boundary; no axiom is removed to bypass it.

Two SPARQL queries accept an explicit record binding: `queries/object-manipulation-targets.rq` and `queries/object-manipulation-monitors.rq`. Example record IRI:

`http://aicor.knowledge/ontologies/object-manipulation-examples.owl#pickupRecord`

## Remaining domain work

Populate sourced grasp/support/contact models; connect capability requirements to verified providers; define quantitative stability, clearance and hold-quality criteria; add detailed grasp/lift/retreat strategy refinements and interval evidence-coverage policies. Concurrent multi-arm manipulation and handover variants remain extensions. The forthcoming mobility, articulated-mechanism, tool and recipe modules can reuse these descriptions. Working projections and CoraPlex execution adapters remain deferred.

## Release verification

102 object-manipulation checks, 14 scoped query cases, 41 import checks, 58 robot-motion regression checks and 93 historical competency assertions passed. The standalone domain and combined assembly classify with no unsatisfiable named classes; generic examples consistency passed. The optional AICOR check remains blocked by the existing HermiT error. Of 60 prior ontology/example files, 59 are byte-identical; only the combined action-knowledge assembly changed to import the new domain.
