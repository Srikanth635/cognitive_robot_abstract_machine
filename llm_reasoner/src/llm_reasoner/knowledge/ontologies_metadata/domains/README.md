# Reusable action domains — 0.20.0

The package now includes seven domains plus two composition modules and one tool-refinement module; see **../compositions/ACTION_RECIPES.md**, **../compositions/TRANSPORT_CARRYING.md** and **../refinements/SPECIALIZED_TOOL_INTERACTIONS.md**. **ARTICULATED_MECHANISMS.md** documents the new revolute/prismatic configuration and latch profiles. **TOOL_INTERACTION_MATERIAL_TRANSFER.md** documents the fifth and sixth modules and **../demo/README.md** introduces the comprehensive illustration.

The package includes **perception/attention** as its fourth domain; see **PERCEPTION_ATTENTION.md**. It includes **mobility/navigation** as its third domain; see **MOBILITY_NAVIGATION.md**. It also includes **object manipulation** as its second domain. See **OBJECT_MANIPULATION.md** for the new module. The robot-motion documentation below describes its 0.13 foundation, retained in this release.

The first domain module is **robot-motion.ttl**, with ontology IRI:

`http://aicor.knowledge/ontologies/robot-motion.owl`

It imports conditions/outcomes, monitoring requirements and model contracts. DUL is loaded transitively through the core. It has no dependency on CoraPlex, SDT, ROS, a particular robot model, a simulator or a live world.

Open this module alone for motion-domain modelling. Open **integrations/all-action-knowledge.ttl** for the combined aspect/domain assembly. The prior **all-action-aspects.ttl** stays domain-independent and unchanged. Registry and local catalogs resolve both assemblies offline. The package now contains 40 registered ontology documents, including seven reusable domain modules.

## What this domain describes

| Motion family | Target structure | Completion requirement in the bounded profile |
|---|---|---|
| Configuration motion | Coordinate specifications, target parameters, error and tolerance parameters | Every coordinate error is within its corresponding tolerance |
| Cartesian positioning | Position parameters, explicit frame, optional orientation target | Endpoint error within tolerance; an orientation request additionally requires an angular error/tolerance pair |
| Path following | Ordered waypoint pose targets | Endpoint and path tracking error within their separate tolerances; orientation requests additionally need an angular metric |

The shared `RobotMotionDescription` specifies a controlled entity description, an action kind, preconditions, a goal, a success criterion, an abstract capability requirement and optional motion bounds. Targets and bounds reuse core `ParameterSpecification`. Configuration targets identify abstract coordinates by key and linear/angular kind. Position targets contain three scalar parameters. Orientation targets describe quaternion XYZW parameters under an explicit convention. No target parameter is a physical joint or measured world coordinate.

Motion error metrics are supplied by a future qualified observer/model adapter. Coordinate errors are nonnegative distance/angular-error metrics. Endpoint error refers to the specific target and controlled part. Tracking error is an aggregate over the requested path, not an endpoint measurement. This module does not compute metrics from robot states, define a path sampling guarantee or infer that a supplied metric was measured correctly.

The six current profile concepts are:

- Joint Repositioning.
- Stowing.
- Gripper Configuration Change.
- Gripper Aperture Change.
- Cartesian Positioning.
- Pose Path Following.

The domain descriptions specialize three reusable classes; the six profile kinds are individuals of `ac:ActionKind`, consistent with the existing core pattern. They do not denote actual robot actions. Profiles may later combine with other domain modules. Closing a gripper is a configuration change; acquiring or maintaining a grasp additionally needs the forthcoming object-manipulation domain.

## Example knowledge and provenance

`examples/robot-motion-profiles.ttl` provides six synthetic records. The joint-repositioning example has one abstract angular coordinate; the other configuration examples have one abstract linear coordinate. These are minimal templates to exercise both coordinate kinds, not claims about the joints any real robot has. Stowing a real arm usually needs a complete bound configuration; the one-coordinate fixture is not a deployable stowing program.

Cartesian/path example targets currently specify position only. Orientation structures are available, and the integrity checker rejects an orientation request without an explicit angular completion metric. Quaternion normalization, orientation error computation and time-varying frame transforms are not implemented. An orientation-aware populated profile remains follow-up work.

Target values, error measurements, tolerances, speed limits and acceleration limits are deliberately unspecified. No robot-specific default or safety threshold is generated. Each example records explicit Proposed claims for its reachable descriptive non-type/non-label links, backed by evidence labelled as synthetic. Claims and records can receive contextual assessments through the existing provenance/context module; no observation or accepted assessment is fabricated here.

A precondition describes the need for coordinate/frame bindings, resource availability and applicable bounds. Its text is not an executable precondition proof. Completion queries inspect supplied errors and tolerances only. They do not establish that preconditions hold or that execution is safe/feasible.

## Bounded authoring and inspection contracts

`robot-motion-shapes.ttl` supplies strict authoring checks in addition to the open-world OWL schema. `tools/robot_motion.py` checks:

- Coordinate target completeness and duplicate coordinate identities/keys.
- Consistent explicit units across a coordinate's target, error and tolerance.
- Known canonical metre/radian identifiers against the coordinate kind. Other units require an external dimension/unit catalog; compatibility is not inferred here.
- Contiguous unique waypoint indices and a common path reference frame. Mixed frames require an explicit transformation contract and are rejected by this first profile.
- Nonnegative finite numerical bounds when provided.
- Explicit endpoint/tracking metric bindings and additional angular metrics when orientation is requested.
- Each required metric pair occurs as a positive conjunct of `post-state error <= task-context tolerance` in the success expression.

This intentionally bounded completion profile accepts conjunctions of scalar comparisons. It does not attempt arbitrary logical entailment, time-indexed temporal logic or pose algebra. Run shapes and integrity validation before reference evaluation.

`assess_completion` uses caller-supplied quantities keyed by `(state scope, parameter IRI)`. It returns True, False or Unknown for completion and always reports that execution/readiness were not assessed. Errors and tolerances must be finite and nonnegative; supplied unit identifiers must match parameter declarations exactly. No conversion occurs. Independent contexts must use isolated input dictionaries and explicit record/context selection; do not merge measurements from separate motion instances.

There is no default trajectory interpolation, timing law, control period or controller. A speed/acceleration bound is a descriptive constraint requiring later planning/control verification. Reaching an endpoint does not demonstrate that a path, speed bound, collision constraint or orientation constraint was satisfied.

## Competency questions

| Question | Coverage |
|---|---|
| What part is intended to move, toward which kind of target? | Explicit controlled-entity and target links; scoped SPARQL query |
| Which coordinates and target/error/tolerance parameters are required? | Configuration target entries |
| Which frame is a Cartesian target expressed in? | Mandatory descriptive frame specification |
| Are waypoint indices complete and do their frames agree? | SHACL plus Python integrity checks |
| Is endpoint arrival enough for path success? | No; a separate tracking comparison is required and tested |
| Does a position-only check establish an orientation target? | No; the authoring profile requires an angular metric |
| What happens if an error or tolerance is missing? | Completion remains Unknown unless another conjunct already establishes False |
| Do these descriptions prove reachability or controller feasibility? | No; these require injected models, bindings and external verification |

## Reproduce validation

With requirements installed, from the package root:

```bash
python tools/validate_robot_motion.py
python tools/reason_robot_motion.py
python tools/validate_imports.py
python tools/review_competency.py
python tools/validate_context.py
```

Java is required for classification/consistency. Domain tests use the core shapes, existing condition/aspect shapes and new domain shapes. Results are in `reports/robot-motion-validation.json` and `reports/robot-motion-reasoning.json`. The latter classifies the domain and combined assembly and separately checks consistency of all examples plus the optional AICOR/CRAM bridge. Full AICOR satisfiability remains outside that consistency check.

`queries/robot-motion-targets.rq` requires an explicit record binding. It returns descriptive target links. Example record IRI:

`http://aicor.knowledge/ontologies/robot-motion-examples.owl#posePathFollowingRecord`

## Historical robot-motion follow-up notes

Develop object manipulation on top of these motion descriptions: reaching to acquire a grasp, establishing a hold, lifting clear of support, maintaining a hold, placement and release. CoraPlex mappings remain a separate adapter. Actual robot coordinate bindings, frame transforms, FK/IK, trajectory generation, collision checking, orientation-aware populated profiles and controller validation are subsequent integration/model work. This release creates no new working projections.

## Current integration boundary

The domain and new assembly classify successfully. The generic combined knowledge/examples consistency check is reported separately from the optional AICOR check. The optional AICOR/CRAM check currently encounters a HermiT `DisjointClasses(SceneNode)` normalization error. Inspection finds eight members in that AICOR RDF list before and after serialization; the precise reason for HermiT's singleton interpretation remains unresolved. No original AICOR axiom was changed or removed to bypass it. Treat optional AICOR compatibility as unverified for this release.

This is recorded as a blocked optional check in `reports/robot-motion-reasoning.json`; it is not counted as a successful consistency result. The standalone motion domain does not import AICOR.

## Release verification summary

58 domain checks, six scoped query cases and 40 import checks passed. The existing 93 competency assertions and 26 context assertions also passed. Both new ontology closures classify with no unsatisfiable named classes; generic knowledge plus examples is consistent. The optional AICOR check remains blocked as documented above. All 56 prior ontology/example files are byte-identical to release 0.12.0.
