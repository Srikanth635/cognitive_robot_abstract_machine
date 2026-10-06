# Flanagan Microworld Enhancement Plan

## Progress

- Completed: Phase 0 through Phase 10
- Current: Plan complete
- Overall: 11 of 11 phases complete
- Last validated: 2026-09-27

## Objective

Enhance the Flanagan microworld so that it provides broad and scientifically
traceable coverage of Flanagan, Bowman, and Johansson's predictive manipulation
framework, supports reliable language-model generation and querying, and admits
no internally contradictory canonical knowledge structures.

The work must remain test-driven. Each defect or missing behavior is first
captured by a focused failing test, followed by the smallest coherent model
change that makes the test pass.

## Phase 0: Establish the scientific contract

### Objective

Define exactly what belongs to the Flanagan framework before changing the
ontology or generated structures.

### Work

Create a claim matrix with one row per relevant paper concept:

```text
paper section
-> claim
-> evidence status
-> microworld construct
-> constraint
-> competency question
-> test
```

Classify claims with an enum such as:

- `EMPIRICAL_FINDING`
- `AUTHOR_HYPOTHESIS`
- `OPEN_QUESTION`
- `ENGINEERING_EXTENSION`

Record these non-negotiable distinctions:

- Physical event versus sensory evidence
- Predicted evidence versus actual evidence
- Phase boundary versus within-phase regulation
- Motor correction versus internal-model update
- Mass versus weight force
- Static property versus current state versus dynamic model
- Sensorimotor representation versus perceptual representation
- Description-level stereotype versus occurrence-level observation

### Exit criteria

- Every major paper section appears in the claim matrix.
- Every existing microworld class is classified as paper-backed or an
  engineering extension.
- Competency questions define what the finished system must answer.
- No extension is implicitly attributed to Flanagan.

## Phase 1: Lock down existing failures

### Objective

Prevent the current contradictory cereal-pouring structure from remaining the
implicit reference model.

### Work

- Add a failing test that evaluates the entire hand-authored structure with the
  real Flanagan acceptance suite.
- Add regression coverage for the currently observed 21 errors and 5 warnings.
- Give every parameter its required participant roles.
- Ensure every continuously regulated phase monitors an appropriate signal.
- Ensure every precondition is established upstream or explicitly marked as a
  world assumption.
- Replace unjustified generic boundary events with meaningful event types.
- Give every control point a justified mismatch policy, or explicitly model why
  no preselected correction exists.
- Make the reference fixture part of the package-level validation path rather
  than testing it only for non-empty collections.

### Exit criteria

```text
SUITE.evaluate(reference_record.scope.fragments) == []
```

The canonical structure has zero acceptance errors and zero unexplained
warnings.

## Phase 2: Correct the foundational ontology

### Objective

Remove the event, control, and quantity contradictions on which later modules
currently depend.

### Event characteristics

Replace mutually exclusive contact, force-dynamic, and control event kinds with
orthogonal event characteristics. A physical event may simultaneously:

- Establish or break contact
- Cross a mechanical threshold
- Attain a controller setpoint

`Liftoff` must express both support-contact break and a load-force threshold
crossing. `Touchdown` must express both support-contact establishment and a force
transition.

### Phase control dimensions

Separate how a phase terminates from how it is regulated internally:

```text
Phase boundary:
    contact event | threshold event | setpoint event | external termination

Within-phase regulation:
    feedforward | continuous feedback | hybrid
```

This must allow a transport phase to be continuously regulated while still
ending at a discrete event.

### Parameter categories

Separate at least:

- `ObjectPropertyEstimate`
- `ContactPropertyEstimate`
- `ObjectStateEstimate`
- `DynamicsEstimate`
- `MotorSetpoint`
- `TaskSetpoint`
- `MechanicalThreshold`
- `SafetyMargin`

Do not force categorical shape, identity, or material regime into a numerical
range.

### Dimension-safe quantities

Represent and validate physical dimensions, including:

- Mass
- Force and weight force
- Torque
- Angle
- Angular rate
- Volume
- Flow rate
- Friction coefficient

### Exit criteria

- Lift-off and touchdown satisfy both contact and threshold competency tests.
- Discrete boundaries coexist with continuous regulation.
- Invalid mass/force assignments are rejected.
- Categorical properties no longer use fabricated numeric ranges.
- Canonical values are dimensionally consistent.

### Implementation record (completed 2026-08-27)

- Added a generic `PhaseBoundaryEventType` layer and independent contact,
  threshold-crossing, control, and external-termination characteristics.
- Made `Liftoff` and `Touchdown` combine contact topology with explicit
  force-threshold crossings.
- Replaced the conflated control regime with `RegulationMode` values for
  feedforward, continuous-feedback, and hybrid regulation, while retaining
  event-based phase boundaries.
- Added distinct estimate, motor/task setpoint, mechanical-threshold, and
  safety-margin categories; categorical values now use concept references.
- Added typed physical dimensions and measurement units, including safe
  unit-to-dimension inference for legacy generated values.
- Added constraints C33-C36 for unit compatibility, exclusive value forms,
  parameter dimensions, and quantitative/categorical form; C13d now enforces
  closure over linked thresholds, setpoints, estimates, and force-coupling
  parameters.
- Updated the canonical pouring structure, query vocabulary, and object
  representation so every declared estimate is connected to its model.

Verification:

- The canonical Flanagan acceptance suite returns no violations.
- The structural ontology validator reports 0 errors. Its 79 V04 findings are
  non-contradictory sibling-overlap notices, including the event-characteristic
  overlaps deliberately retained by this phase.
- The definitive regression scope reports 246 passed and 1 skipped.

## Phase 3: Represent the complete predictive-control loop

### Objective

Make a sensorimotor control point an operational comparison rather than a label
holder.

### Work

Add concepts equivalent to:

- `SensoryEventPrediction`
- `SensoryObservationType`
- `PredictionComparison`
- `PredictionError`
- `PredictionErrorKind`
- `CorrectiveAction`
- `ModelUpdate`
- `InitialStateEstimate`

Cover at least these prediction-error kinds:

- `ABSENT`
- `UNEXPECTED`
- `EARLY`
- `LATE`
- `MAGNITUDE_MISMATCH`
- `DIRECTION_MISMATCH`
- `SPATIAL_MISMATCH`

A sensory prediction must identify:

- Modality
- Predicted feature or event
- Causing mechanical event
- Expected timing window
- Expected location or participants
- Optional magnitude range and tolerance
- Model that generated the prediction

A comparison must support both outcomes:

```text
match
-> confirm subgoal
-> provide the next-phase state estimate
-> permit the phase transition

mismatch
-> classify the error
-> invoke a corrective action
-> optionally repeat or replan the phase
-> update the relevant representation
```

Add paper-grounded reference scenarios:

- Heavy object: absent lift-off, increased load force, repeated lifting, and
  updated weight estimate
- Light object: early lift-off and reduced future force rate
- Unexpected slip: rapid grip-force increase and updated friction estimate
- Unexpected contact: protective correction

### Exit criteria

- The heavy-object example is expressible without interpreting free-text labels.
- A transition can depend on comparison success rather than event occurrence
  alone.
- Corrections identify their trigger, target setpoint, direction, and intended
  effect.
- Model updates identify the estimate or representation they change.

### Implementation record (completed 2026-08-27)

- Replaced label-only control points with explicit sensory predictions,
  observation types, comparisons, typed prediction errors, corrective actions,
  recovery policies, model updates, and initial-state estimates.
- Added temporal windows and timing units, typed comparison outcomes, all seven
  required prediction-error kinds, and explicit correction and estimate-update
  directions.
- Made successful comparisons gate reciprocal phase transitions and provide
  the next phase's initial-state estimate.
- Added normal contact, lift-off, and flow comparisons plus paper-grounded heavy
  object, light object, unexpected-slip, and unexpected-contact mismatch paths.
- Added constraints C37-C50 for prediction shape, dimensional consistency,
  observation/error coherence, transition gating, ownership closure, and
  match/mismatch completeness.
- Extended indexing and typed correction queries over the predictive-control
  graph, and extended generation-schema coverage for the new contracts.

Verification:

- The canonical Flanagan acceptance suite returns no violations.
- The structural ontology validator reports 0 errors.
- The definitive regression scope reports 280 passed and 1 skipped.

## Phase 4: Rebuild internal models and object dynamics

### Objective

Represent what produces predictions, how object and limb dynamics differ, and
how learned representations change.

### Work

Represent this composition explicitly:

```text
desired state
-> inverse model
-> motor command
-> efference copy
      +
object-dynamics representation
-> forward model
-> predicted mechanical and sensory consequences
```

Add concepts such as:

- `EfferenceCopy`
- `ObjectDynamicsModel`
- `LimbDynamicsModel`
- `CoordinateFrame`
- `RepresentationPurpose`
- `SensorimotorMemory`
- `RepresentationUpdate`

Coordinate frames should include:

- `OBJECT_CENTERED`
- `ARM_CENTERED`
- `HAND_CENTERED`
- `GAZE_CENTERED`

Representation purposes should distinguish:

- `SENSORIMOTOR_CONTROL`
- `PERCEPTUAL_JUDGMENT`

Add reference scenarios for:

- Loads applied to the object versus equivalent loads applied to the arm
- Object-specific adaptation disappearing when the object is released
- Object rotation versus hand rotation
- Accurate motor scaling alongside the persistent size-weight illusion

### Exit criteria

- An object-dynamics model explicitly uses an efference copy.
- Object and limb dynamics can be queried independently.
- Sensorimotor and perceptual weight representations can disagree without
  contradiction.
- Every estimate belongs to a representation and declares its coordinate frame
  and purpose where relevant.

### Implementation record (completed 2026-08-28)

- Added an action-neutral `object_dynamics.py` vocabulary, separate from the
  pouring extension, for desired states, motor commands, efference copies,
  object and limb dynamics, representation purposes and coordinate frames,
  sensorimotor memories, updates, perturbations, transfer, and dissociation.
- Added a separate `object_dynamics_constraints.py` module with composition,
  ownership, purpose-separation, update, transfer, model-type, role-coverage,
  and root-closure constraints.
- Made the desired-state-to-prediction composition explicit and connected its
  forward model to the canonical predictive-control consequences.
- Added canonical object-versus-arm load, object-versus-hand rotation,
  release-dependent memory, and size-weight dissociation scenarios.
- Registered object dynamics as its own capability and module dependency, and
  extended typed queries and generation schemas without moving generic
  dynamics concerns into the pouring module.

Verification:

- The canonical Flanagan acceptance suite returns no violations.
- Phase 4 acceptance and shared generation-contract tests report 93 passed.
- The definitive regression scope reports 309 passed and 1 skipped.
- Targeted Ruff checks and Python compilation pass for every Phase 4 file.

## Phase 5: Extract and correct prehensile manipulation

### Objective

Move general grasping, lifting, and force control out of the pouring extension.

### Work

Create a `prehension.py` module containing:

- Effector, grasp-site, manipulandum, and support roles
- Contact establishment and release
- Force loading
- Lift-off and touchdown
- Grip and load forces
- Force-rate setpoints
- Slip threshold and safety margin
- Grip-load coupling
- Corrective grip responses

Use clear phase concepts such as:

```text
Reach
ContactEstablishment
ForceLoading
Lift
Transport
Replace
Unload
ContactRelease
```

Optional hold, alignment, and withdrawal phases must not be mandatory for every
prehensile action.

Correct the release sequence:

- `UnloadPhase` ends when object weight has transferred to the support and the
  forces reach their intended low state.
- `ContactReleasePhase` ends at effector-object contact break.
- A separate `WithdrawPhase` follows only when the task requires it.

Represent grip-load coupling as a functional relationship involving:

- Predicted load
- Friction estimate
- Safety margin
- Grip-force response
- Temporal coupling

Do not reduce the relationship to an unexplained constant ratio.

### Exit criteria

- Plain grasping no longer loads pouring-specific vocabulary.
- Lowering remains coupled until touchdown.
- Unloading, contact release, and withdrawal are unambiguous.
- Normal lift, slip correction, transport coupling, and anticipatory impact grip
  all have reference tests.

### Implementation record (completed 2026-08-28)

- Added an action-neutral `prehension.py` module for prehensile roles, contact
  and force events, phase concepts, force setpoints and thresholds, object
  estimates, reactive grip correction, and anticipatory grip adjustment.
- Added a separate `prehension_constraints.py` module for coupling coherence,
  lowering, unloading, contact release, optional withdrawal, typed grip
  responses, anticipatory touchdown context, and root closure.
- Made `PrehensionPhaseStructure` extend `ObjectDynamicsPhaseStructure`, and
  made `PouringPhaseStructure` extend prehension while retaining only
  substance-transfer vocabulary in `pouring.py`.
- Registered prehension as its own capability provider, so plain grasping loads
  core, object dynamics, and prehension without loading pouring.
- Replaced the ratio-only grip-load relation with represented load, friction
  estimate, derived safety margin, grip-force response, temporal coupling, and
  active-phase context.
- Split force unloading, contact release, and optional withdrawal into distinct
  phases and boundaries; lowering remains coupled through touchdown.
- Added canonical normal lift, transport coupling, slip correction, support
  transfer, and anticipatory touchdown grip scenarios with typed queries.

Verification:

- The canonical Flanagan acceptance suite returns no violations.
- Focused Phase 5 tests report 21 passed.
- The definitive regression scope reports 330 passed and 1 skipped.
- Targeted Ruff checks and Python compilation pass for every Phase 5 file.

## Phase 6: Rebuild pouring as a true extension

### Objective

Keep only substance-transfer-specific knowledge in `pouring.py` and compose it
with the prehension module when needed.

### Work

Retain pouring-specific concepts such as:

- Source container
- Target receptacle
- Poured substance
- Pouring lip
- Alignment
- Flow onset and cessation
- Flow regulation
- Target quantity
- Spill conditions
- Changing center of mass

Add explicit controlled variables:

- Flow-rate setpoint
- Transferred-quantity setpoint
- Tilt angle and rate
- Alignment error
- Remaining source quantity
- Target fill state

A continuous prediction must identify its signal, predicted trajectory,
tolerance, generating model, and comparison target rather than encoding these in
its label.

Distinguish material regimes, at least:

- Liquid flow
- Granular flow
- Cohesive or highly viscous flow where justified

The cereal reference fixture must use granular-flow properties. A separate
water-pouring fixture should exercise viscosity and slosh.

Support variants including:

- Pouring from an already-held container
- Pouring without transporting first
- Stopping at a requested quantity
- Stopping because the source is empty
- Returning or not returning the source
- Replacing on the original or a different support

### Exit criteria

- Water and cereal fixtures use appropriate and distinct dynamics.
- Flow predictions specify real sensory signals and controlled variables.
- Prehensile phases are composed when needed rather than duplicated.
- Valid pouring variants do not have to invent irrelevant phases.

### Implementation record (2026-09-25)

- Moved action-neutral auditory and continuous-prediction vocabulary into
  `flanagan.core`; pouring now owns only substance-transfer concepts.
- Added typed trajectories and complete continuous-comparison contracts covering
  the signal, expected trajectory, tolerance, model, observation target,
  controlled variable, and active phase.
- Added mass- and volume-aware flow-rate, transferred-quantity, remaining-source,
  target-fill, alignment-error, tilt, and centre-of-mass representations.
- Added liquid, granular, and cohesive regimes with compatibility constraint
  `C57`; the cereal fixture now uses granular dynamics and mass units, while the
  water fixture uses viscosity, slosh, and volume units.
- Added typed termination reasons and constraint `C58`, which requires a matching
  declared boundary event for target-quantity, source-empty, external-request, or
  failure termination.
- Relaxed mandatory phase coverage to the transfer core and retained the full
  ordering constraint for whichever optional prehension, transport, return, or
  replacement phases a task variant actually includes.
- Advanced the core and pouring module versions to `0.3.0` and `0.4.0` so older
  cached hypotheses cannot be mistaken for the expanded schema.

Verification:

- The isolated `llm_reasoner` suite reports 135 passed.
- The relevant Flanagan and hypothesis-knowledge integration suite reports 56
  passed.
- Python compilation and targeted Ruff import/error checks pass for every Phase 6
  file.
- `scripts/format_docstrings.py` completed on every modified Python file.

## Phase 7: Add the remaining Flanagan coverage

### Objective

Cover the review beyond hand-force execution while preserving scientific status
and modularity.

### Gaze control

Represent:

- Gaze target or contact location
- Proactive versus reactive gaze
- Gaze lead relative to hand arrival
- Fixation until subgoal completion
- Visual monitoring of contact events
- Multimodal alignment at control points

### Action observation

Represent:

- Actor and observer
- Observed-hand visibility
- Predicted future contact site
- Modalities available to the observer
- Inference about hidden object properties
- The hypothesis status of actor-observer matching

### Visuomotor learning

Represent:

- Exploration
- Skill acquisition
- Skill refinement
- Reactive gaze becoming proactive
- Novel-rule discovery versus adaptation of an existing mapping
- Sensorimotor-map update

### Sensory attenuation

Represent separately from action control:

- Predicted self-generated contact
- Temporal attenuation window
- Attenuated perceptual response
- External versus self-generated event

The proposed relationship between sensory attenuation and manipulation control
must remain marked as unresolved rather than asserted as an established shared
mechanism.

### Exit criteria

- Every major paper section has a model construct and competency test.
- Empirical findings, author hypotheses, and open questions remain distinct.
- Optional modules are loaded only for tasks or questions that need them.

### Implementation record (2026-09-25)

- Added separate `gaze.py`, `observation.py`, `learning.py`, and `perception.py`
  modules so each scientific topic retains one clear responsibility.
- Added shared scientific-status vocabulary distinguishing empirical findings,
  supported interpretations, author hypotheses, open questions, and engineering
  extensions.
- Represented proactive and reactive gaze, contact targets, gaze lead,
  subgoal-bound fixation, contact monitoring, and multimodal alignment.
- Represented distinct actor and observer roles, hand visibility, future-contact
  predictions, available observer modalities, hidden-property inference, and the
  hypothesis status of actor-observer prediction matching.
- Represented exploration, acquisition, and refinement; reactive-to-proactive
  gaze development; novel-rule discovery versus mapping adaptation; and
  sensorimotor-map updates.
- Represented self-generated versus external events, predicted contact,
  attenuation windows and responses, while keeping the proposed relationship to
  action control explicitly open.
- Added local constraint groups C59--C65 and composed them through the aggregate
  Flanagan constraint registry.
- Registered the four topics as optional knowledge modules with dependency-aware
  capability selection. Existing pouring selection does not load them, while an
  explicit action-observation request loads only its gaze dependency.
- Exposed each new root type independently to generation-schema discovery and
  advanced the core vocabulary module to version `0.4.0`.
- Verified 145 isolated `llm_reasoner` tests and 56 relevant integration tests.
  Python compilation, targeted Ruff checks, and repository docstring formatting
  also pass for the Phase 7 files.

## Phase 8: Strengthen constraints

### Objective

Make semantically contradictory generated graphs impossible to admit.

### Work

Add constraints requiring that:

- Every normal phase boundary has an appropriate control point or an explicit
  exemption.
- Every prediction refers to a sensory event and a generating model.
- Every comparison refers to predicted and actual evidence types.
- Every correction names its triggering error and intended effect.
- Every model update identifies what it updates.
- Every continuously regulated phase has a controlled variable, monitored
  signal, predicted trajectory, and setpoint.
- Every parameter has roles, phases, units, and correct dimensions.
- Every object estimate belongs to a representation.
- Contact changes agree with phase conditions.
- Failure events never gate normal success transitions.
- Engineering extensions carry explicit provenance.
- Canonical fixtures use the most specific available concepts.

Test contradictory combinations rather than only missing required fields.

### Exit criteria

- Canonical fixtures have zero errors and zero unexplained warnings.
- Deliberately malformed fixtures exercise every constraint.
- A schema-valid but scientifically contradictory hypothesis is rejected.

### Implementation record (2026-09-25)

- Added explicit boundary-control exemptions and constraint `C66`, so each
  successful phase boundary now has a sensorimotor control point or a typed,
  justified exemption. Constraint `C69` rejects unused and failure-boundary
  exemptions.
- Added evidence-compatibility constraint `C67`, which requires matching
  modality and participant roles and matching event or feature identity while
  preserving explicitly classified unexpected-event mismatches.
- Added the abstract `ScientificAssertion` contract and typed
  `ScientificProvenance`; engineering extensions without provenance are now
  rejected by `C68` without forcing paper-backed claims to invent sources.
- Made each parameter's active phase required and added module-local closure
  checks for continuous predictions, controlled setpoints, comparison targets,
  and generating models.
- Added `POUR-C66`, requiring every continuous-feedback or hybrid phase to bind
  a monitored signal, predicted trajectory, and controlled setpoint.
- Added `PRE-C66`, rejecting contact-establishment and touchdown events that
  break contact, and contact-break and liftoff events that establish it.
- Qualified extension-owned constraint identifiers by module, eliminating
  ambiguous diagnostics while retaining the legacy check number.
- Completed the cereal and water reference fixtures with continuous grip,
  transport, flow-cessation, and release contracts plus explicit fixture-scope
  boundary exemptions. Both canonical structures now validate without
  diagnostics.
- Advanced changed module versions so cached generated hypotheses cannot reuse
  the pre-hardening schema or constraint contract.

Verification:

- The isolated `llm_reasoner` suite reports 158 passed.
- The relevant Flanagan and hypothesis-knowledge integration suite reports 117
  passed and 1 skipped.
- Python compilation and targeted Ruff import/error checks pass for every
  modified Phase 8 Python file.
- `scripts/format_docstrings.py` completed on every modified Python file.

## Phase 9: Improve language-model generation and querying

### Objective

Ensure that language models can generate every represented aspect and that
consumers can retrieve the knowledge through typed questions.

### Prompt and schema work

- Keep prompt-facing docstrings limited to semantic contracts.
- Remove implementation history and rejected-design comparisons.
- Remove hardcoded quantitative priors from the generic prompt.
- Explain the distinction between paper-backed claims and engineering
  extensions.
- Select the root structure and capability modules from the task profile rather
  than hardcoding pouring.
- Require the most-specific available type and explicit scientific status.

### Query work

Add typed questions equivalent to:

```text
expected_events_at(control_point)
comparison_at(control_point)
prediction_errors_for(event)
corrections_for(error)
model_updates_for(error)
next_phase_after(comparison)
dynamics_model_for(phase)
force_coupling_for(phase)
continuous_predictions_for(phase)
gaze_target_for(phase)
representation_for(property, purpose)
```

Keep generic path following as an escape hatch, not the primary access mechanism
for central Flanagan concepts.

Use mock-generated payloads for CI tests. Live model tests must be skipped when
approved credentials are unavailable.

### Exit criteria

- Every competency question has a typed query method.
- Mock language-model outputs cover normal, variant, and failure cases.
- Prompt and schema tests prevent unsupported concepts and fields.
- Generated hypotheses pass the scientific constraint suite before admission.

### Completed implementation

- Added task-profile generation roots for action-neutral, prehension, and
  pouring hypotheses while preserving capability-based module closure. The
  generator now derives its root from the selected profile unless a specialized
  caller supplies an explicit override.
- Removed the viewer's pouring-root constant and the generic prompt's numeric
  mass example. Added explicit prompt guidance for the paper-backed,
  interpretation, author-hypothesis, open-question, and engineering-extension
  statuses while retaining the most-specific-type requirement.
- Added typed, cited queries for control-point events and comparisons,
  prediction errors, corrections, model updates, successful next phases,
  phase dynamics, grip-load coupling, continuous predictions, and gaze targets.
  The existing representation-purpose query and generic path-following escape
  hatch remain available.
- Added offline model payload tests for normal decoding, task-profile variants,
  structural repair, and scientific-constraint repair. Admission is asserted
  only after the scientific suite accepts the repaired graph.
- Confirmed that schema derivation exposes only the first semantic paragraph of
  class documentation, so implementation history is not prompt-facing.

Verification:

- The isolated `llm_reasoner` suite reports 163 passed.
- The Flanagan query and hypothesis-knowledge regression subset reports 58
  passed.
- The viewer generation regression suite reports 20 passed and 1 skipped; the
  skipped test requires an approved live model credential.
- Targeted Ruff import/error checks and Python compilation pass for every
  modified Phase 9 Python file.
- The required formatter script was invoked, but the active environment lacks
  its `docformatter` executable. The same repository-configured `docformatter`
  and active-environment Black steps were run directly and completed.

## Phase 10: Migration and certification

### Objective

Finish with one coherent, versioned, documented, and fully tested microworld.

### Work

- Replace placeholder `example.org` identifiers with a stable project namespace.
- Bump module versions and invalidate incompatible cached hypotheses.
- Migrate or deliberately expire stored records from the old model.
- Move canonical fixtures into the package that owns their constraints.
- Remove the obsolete Turtle fixture loader if its documented fixtures no longer
  exist and no supported workflow uses it.
- Document architecture, vocabulary, scientific scope, extensions, and examples
  in the package README.
- Run `scripts/format_docstrings.py` on every modified file.
- Run targeted and full pytest suites with the active project interpreter.

### Final definition of done

- Every in-scope paper claim is traceable from source to construct, constraint,
  query, and test; every deliberate exclusion is traceable to the documented
  scope boundary.
- Every engineering addition is explicitly identified.
- Every canonical structure has zero validation violations.
- No unit, event, phase, control, or representation contradiction remains.
- All major paper scenarios are queryable without interpreting free-text labels.
- Generation and query tests cover every competency question.
- Core prediction, object dynamics, prehension, pouring, gaze, learning,
  observation, and perception remain functionally separate modules.

### Completed implementation

- Replaced all Flanagan placeholder IRIs with the project-owned namespace root
  `http://www.ease-crc.org/ont/llm_reasoner/flanagan/` and distinct module
  suffixes defined in one namespace module.
- Certified all eight microworld modules at version `1.0.0` and the generation
  contract at `generation/v3`. Cache lookup now considers both module and prompt
  versions, retains incompatible history, and reports it as stale.
- Moved the cereal and water reference builders into the Flanagan package. The
  former viewer path is a compatibility import only, preserving existing
  consumers without making the microworld depend on the viewer.
- Removed the unused Turtle fixture loader after verifying that neither its
  documented fixtures nor a supported caller remained.
- Replaced the obsolete integration README with the current architecture,
  vocabulary ownership, scientific-status rules, explicit engineering
  extensions, migration behavior, query surface, and a validated example.
- Updated the scientific contract with final functional dependencies and a
  traceability row for every FJ claim. FJ-013 is explicitly excluded because
  afferent first-spike coding is outside the declared task-level reasoning scope.
- Added final certification tests for namespaces, fixture ownership and
  acceptance, obsolete-loader removal, version migration, and prompt-aware
  handling of stale and refuted records through both cache entry points.

Verification:

- The isolated `llm_reasoner` suite reports 173 passed.
- The Flanagan and hypothesis-knowledge integration suite reports 122 passed and
  1 skipped; the skipped test requires an approved live model credential.
- Both package-owned canonical structures produce zero acceptance diagnostics.
- Targeted Ruff import/error checks, Python compilation, and diff whitespace
  checks pass for the modified Phase 10 files.
- The required formatter script was invoked, but the active environment lacks
  its `docformatter` executable. The repository-configured `docformatter` and
  active-environment Black commands were run directly on all modified Python
  files.

## Plan completion

All planned phases are implemented. Further work should begin as a separately
scoped scientific extension or defect fix so the certified namespace, module
versions, and traceability contract change deliberately.
