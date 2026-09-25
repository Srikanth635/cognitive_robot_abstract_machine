# Flanagan Microworld Scientific Contract

## Status

This document is the Phase 0 authority for correcting and extending the
Flanagan microworld.

- Roadmap phase: 0 of 10, first of 11 phases
- Phase status: complete; the exit review at the end of this document is
  satisfied
- Production-code impact: none
- Ontology migration impact: none

Phase 0 defines scientific scope, semantic distinctions, functional module
ownership, competency questions, and the status of the existing vocabulary.
Later phases may change code only in accordance with this contract or by
explicitly revising the contract first.

## Scientific authority

The primary source for this contract is:

J. Randall Flanagan, Miles C. Bowman, and Roland S. Johansson,
"Control strategies in object manipulation tasks," Current Opinion in
Neurobiology, volume 16, pages 650-659, 2006,
doi:10.1016/j.conb.2006.10.005.

The article is a review and theoretical synthesis. It reports experimental
findings, interprets those findings through predictive control, proposes the
sensorimotor-control-point framework, and identifies unresolved questions.
Those epistemic categories must not be flattened into equally certain facts.

Primary experimental papers cited by the review should be consulted when a
later implementation needs a quantitative relationship, latency, coordinate
frame, or experimental boundary that the review only summarizes. Contemporary
scientific validity beyond the 2006 review requires a separate literature
review; this contract first establishes fidelity to the named work.

## Evidence status

Every scientific claim and every future source-backed construct must carry one
of these statuses:

| Status | Meaning |
|---|---|
| EMPIRICAL_FINDING | The reviewed experiments directly report the behavior or dissociation. |
| SUPPORTED_INTERPRETATION | The authors infer a control organization or representation from converging findings. |
| AUTHOR_HYPOTHESIS | The authors explicitly propose a mechanism or broader functional role that is not yet established. |
| OPEN_QUESTION | The paper states that the mechanism, representation, or relationship remains unresolved. |
| ENGINEERING_EXTENSION | The construct is useful for robotic knowledge or a task extension but is not a claim made by the paper. |

Evidence status describes scientific support, not software maturity or
confidence in a generated instance.

## Scope

The Flanagan microworld represents generic, description-level knowledge about
predictive object manipulation. It does not represent a particular execution,
observed sensory sample, robot, container, or numerical measurement unless a
separate occurrence-level model binds the generic description to the world.

The full target scope covers:

- Action phases, subgoals, and phase boundaries
- Mechanical contact and force-related events
- Predicted and actual multimodal sensory events
- Sensorimotor control points
- Continuous prediction and feedback within phases
- Corrective actions and representation updates
- Internal models of object and limb dynamics
- Prehensile force control
- Gaze control
- Action observation
- Visuomotor learning
- Sensory attenuation around predicted self-contact
- Pouring as an explicitly identified engineering application

Neural anatomy, detailed receptor physiology, and a general ontology of every
human motor behavior are outside the immediate microworld scope. Latencies and
neural codes may be represented as evidence or performance constraints when
needed, but they do not turn this package into a neural simulation.

## Non-negotiable semantic distinctions

These distinctions constrain every later class and relation.

| Distinction | Contract |
|---|---|
| Mechanical event versus sensory event | A mechanical event occurs in the physical interaction. A sensory event is modality-specific evidence caused by or associated with that event. They are never the same individual merely because their timing coincides. |
| Sensory event versus afferent signal | A task-level sensory event describes what is detected. An afferent signal describes its neural encoding. A modality name alone is neither a complete event nor a complete signal. |
| Prediction versus observation | A prediction is an expectation held before evidence is registered. An observation is actual evidence. They may agree or disagree and must remain independently representable. |
| Subgoal versus evidence of attainment | A subgoal is a desired world condition. A contact or sensory event supplies evidence that the condition was attained; it is not automatically the condition itself. |
| Event occurrence versus comparison success | Registering an event does not prove that its timing, location, magnitude, direction, or modality pattern matched the prediction. |
| Phase boundary versus within-phase regulation | A phase may end at a discrete event while its forces, pose, or sensory predictions are continuously regulated. These are orthogonal dimensions. |
| Feedforward command versus feedback correction | A command prepared from a prior estimate is different from a correction released after a prediction error. |
| Corrective action versus model update | A correction protects or completes the current action. A model update changes future predictions. One mismatch may cause either or both. |
| Mass versus weight force | Mass is measured in kilograms. Weight is a force measured in newtons. Names, dimensions, and equations must agree. |
| Object property versus contact property | Mass and shape are object properties. Friction is a property of a contact interface involving skin or effector, object surface, and conditions. |
| Property versus state versus dynamics | Shape is comparatively stable, fill level is a current state, center-of-mass shift is a predicted change, and a force-to-motion mapping is a dynamics model. |
| Motor setpoint versus task setpoint | Grip force and tilt rate are motor-control targets. Desired transferred volume is a task or controlled-variable target, not a motor command. |
| Internal model versus represented content | A forward or inverse model performs a predictive/control function. An object representation supplies content used by such a model. They are not presumed disjoint. |
| Object dynamics versus limb dynamics | Equivalent loads applied to a held object and directly to the arm may be controlled differently and require independently queryable representations. |
| Sensorimotor versus perceptual representation | A property estimate used to control force may differ from a conscious perceptual judgment without contradiction. |
| Description versus occurrence | A generic prototype states stereotypical structure and ranges. An occurrence supplies actual participants, values, observations, and timestamps. |
| Actor versus effector | The actor controls the action. A hand or gripper is an effector role. They are related but not interchangeable. |
| Paper-backed construct versus extension | Flow control, target volume, and other pouring machinery must be labeled as engineering extensions unless independently sourced. |

## Functional module contract

The package must preserve one-way dependencies. The action-neutral core must
never import an action-specific module.

| Module | Owns | May depend on | Must not own |
|---|---|---|---|
| core.py | Generic roles, role slots, phases, subgoals, world conditions, physical events, phase boundaries, and task structures | Ontological infrastructure only | Grip forces, pouring, gaze, observer behavior, or task-specific phases |
| prediction.py | Sensory predictions, observations, comparisons, prediction errors, corrections, state estimates, and generic predictive models | core.py | Object-specific physics, grasp-specific control, or pouring |
| object_dynamics.py | Object and limb dynamics models, efference copy use, coordinate frames, representation purpose, memories, and updates | core.py and prediction.py | Grasp phase taxonomies or pouring flow phases |
| prehension.py | Effector and support roles, grasp/contact phases, lift-off, touchdown, slip, grip/load forces, safety margins, and coupling | core.py, prediction.py, and object_dynamics.py | Source/target vessels, flow, gaze, or perception experiments |
| gaze.py | Gaze targets, fixation, proactive/reactive gaze, and visual monitoring of contact locations | core.py and prediction.py | Actor-observer matching or general skill-stage models |
| observation.py | Actor/observer relations, observer-accessible modalities, observed-hand conditions, and action-understanding predictions | core.py, prediction.py, and gaze.py | Execution-specific force control |
| learning.py | Exploration, acquisition, refinement, mapping discovery, adaptation, and representation updates across trials | core.py, prediction.py, and object_dynamics.py where needed | Pouring-specific flow laws |
| perception.py | Sensory attenuation, self-generated contact prediction, perceptual representations, and unresolved links to action control | core.py and prediction.py | Normal phase-transition control |
| pouring.py | Source, target, substance, lip, flow, alignment, transferred quantity, material regime, and spill-specific knowledge | core.py, prediction.py, object_dynamics.py, and prehension.py | Generic prehension, generic auditory signals, or generic continuous prediction |
| registry.py | Capability-to-module selection and versioned dependency closure | Module metadata | Domain classes or task execution logic |

Constraint ownership must follow the same separation. The final acceptance suite
may aggregate constraints, but task-neutral constraints, prehension constraints,
and pouring constraints must remain separately identifiable and independently
testable.

Dependency direction:

    core
      |
      v
    prediction
      |
      v
    object_dynamics
      |
      v
    prehension
      |
      v
    pouring

Gaze, observation, learning, and perception form optional side capabilities.
None may introduce a reverse dependency into core, prediction, object dynamics,
or prehension.

## Paper claim matrix

Coverage values describe the current package before Phase 1.

| Claim | Paper section | Claim summary | Status | Required representation | Current coverage | Planned owner |
|---|---|---|---|---|---|---|
| FJ-001 | Abstract and introduction, pages 650-651 | Skilled manipulation depends centrally on organization and prediction rather than unusually fast effectors or feedback alone. | SUPPORTED_INTERPRETATION | Predictive phase-control architecture | Partial | core, prediction |
| FJ-002 | Abstract, page 650 | Successful manipulation predicts both required motor commands and their sensory consequences. | SUPPORTED_INTERPRETATION | Desired states, command prediction, forward sensory prediction | Partial | prediction |
| FJ-003 | Sensorimotor control points, pages 650-651 | Manipulation is organized into action phases usually bounded by task-relevant mechanical contact events that serve as subgoals. | AUTHOR_HYPOTHESIS | Phase, subgoal, mechanical boundary, contact change | Covered structurally, but overly universal | core |
| FJ-004 | Sensorimotor control points, pages 650-651 | Contact events produce distinct tactile, proprioceptive, visual, and auditory sensory events or afferent signatures. | EMPIRICAL_FINDING | Physical event, sensory-event type, modality, causal association | Partial; modalities exist but sensory events do not | core, prediction |
| FJ-005 | Sensorimotor control points, page 651 | Comparing predicted and actual sensory events monitors progress and supports corrective action. | SUPPORTED_INTERPRETATION | Prediction, observation, comparison, error, correction | Partial label-level representation | prediction |
| FJ-006 | Sensorimotor control points, pages 651-652 | Control points may monitor multimodal error patterns, align modalities for learning, and uphold sensorimotor correlations. | AUTHOR_HYPOTHESIS | Multimodal comparison and alignment relations | Missing | prediction, learning |
| FJ-007 | Sensorimotor control points, pages 651-652 | Predicted sensory consequences can provide initial state information for the next phase. | AUTHOR_HYPOTHESIS | Initial-state estimate linked to transition target | Missing | prediction |
| FJ-008 | Figure 1 and object transport discussion, pages 651 and 655 | Within a phase, an object model combined with an efference copy predicts load so grip force can be adjusted. | SUPPORTED_INTERPRETATION | Efference copy, object-dynamics model, load prediction, grip command | Partial and disconnected | prediction, object_dynamics, prehension |
| FJ-009 | Prediction and rapid updating, page 652 | Visual and haptic size, shape, and identity cues plus memory predict weight, mass distribution, and stable fingertip forces. | EMPIRICAL_FINDING | Cue, memory, property estimate, force scaling | Partial | object_dynamics, prehension |
| FJ-010 | Prediction and rapid updating, page 652 | Early or absent expected lift-off produces learned force corrections and updates the weight representation. | EMPIRICAL_FINDING | Timing error, absence error, correction, model update | Missing beyond one correction label | prediction, object_dynamics, prehension |
| FJ-011 | Prediction and rapid updating, page 652 | Common loads may update in one trial, while misleading cues or unusual dynamics can require repeated trials. | EMPIRICAL_FINDING | Update timescale, trial context, misleading-cue condition | Missing | learning, object_dynamics |
| FJ-012 | Prediction and rapid updating, page 652 | Tactile ensembles rapidly encode friction, contact shape, and force direction and begin influencing force output after about 70 ms. | EMPIRICAL_FINDING | Tactile feature, latency evidence, contact property | Missing beyond generic tactile modality | prediction, prehension |
| FJ-013 | Prediction and rapid updating, page 652 | Relative first-spike timing provides a rapid distributed temporal code for contact information. | EMPIRICAL_FINDING | Evidence annotation for afferent coding | Missing and optional for task reasoning | prediction |
| FJ-014 | Control of gaze, pages 652-653 | Gaze moves proactively to successive contact locations and usually remains until the current phase goal is completed. | EMPIRICAL_FINDING | Gaze target, lead relation, fixation-until condition | Missing | gaze |
| FJ-015 | Control of gaze, page 653 | Gaze may capture visual contact events, monitor progress, and align them with tactile, auditory, and proprioceptive events. | AUTHOR_HYPOTHESIS | Visual monitoring and multimodal alignment | Missing | gaze, prediction |
| FJ-016 | Action observation, pages 653-654 | Observers of familiar manipulation show proactive, actor-like gaze to forthcoming contact sites. | EMPIRICAL_FINDING | Observer, observed action, predicted contact target | Missing | observation, gaze |
| FJ-017 | Action observation, pages 653-654 | Observer gaze becomes reactive when the actor's hand is hidden and becomes proactive once an ambiguous target is identified. | EMPIRICAL_FINDING | Visibility condition, target uncertainty, gaze mode | Missing | observation, gaze |
| FJ-018 | Action observation, page 653 | Actors and observers may be linked through common sensory predictions and covertly matching action plans. | AUTHOR_HYPOTHESIS | Shared-prediction relation with explicit hypothesis status | Missing | observation |
| FJ-019 | Visuomotor learning, pages 653-654 | A novel action-to-cursor mapping can show exploration, skill acquisition, and skill refinement stages. | EMPIRICAL_FINDING | Learning stage and stage order | Missing | learning |
| FJ-020 | Visuomotor learning, page 654 | Gaze changes from reactive cursor chasing to proactive target-directed control as skill develops. | EMPIRICAL_FINDING | Stage-dependent gaze mode | Missing | learning, gaze |
| FJ-021 | Visuomotor learning, page 654 | Reactive gaze during exploration may help construct gaze-centered sensorimotor mappings. | AUTHOR_HYPOTHESIS | Mapping formation and gaze-centered frame | Missing | learning, gaze |
| FJ-022 | Visuomotor learning, page 654 | Discovering novel mapping rules differs from adapting an already acquired mapping. | SUPPORTED_INTERPRETATION | Learning-process kind | Missing | learning |
| FJ-023 | Sensory cancellation, pages 654-655 | Perceptual attenuation is centered on a predicted self-generated contact event and is not caused by movement alone. | EMPIRICAL_FINDING | Self-generated event, predicted timing, attenuation window | Missing | perception |
| FJ-024 | Sensory cancellation, page 655 | Whether sensory attenuation and action-control prediction use the same neural mechanism is unknown. | OPEN_QUESTION | Explicit unresolved relation | Missing | perception |
| FJ-025 | Object dynamics, pages 655-656 | Grip force is scaled to expected weight and coupled to movement-dependent loads across elastic, viscous, and inertial dynamics. | EMPIRICAL_FINDING | Dynamics kind, load prediction, grip-load coupling | Partial | object_dynamics, prehension |
| FJ-026 | Object dynamics, page 655 | Grip force responds to loads applied to the held object but not identically to equivalent loads applied directly to the arm. | EMPIRICAL_FINDING | Load attachment site, object/limb dissociation | Missing | object_dynamics, prehension |
| FJ-027 | Object dynamics, pages 655-656 | Object-dynamics models can be recruited and de-recruited with grasp and release separately from limb-dynamics control. | SUPPORTED_INTERPRETATION | Model activation context and object/limb model distinction | Missing | object_dynamics |
| FJ-028 | Object dynamics, page 656 | Prior fingertip actions may alter later grip force without altering load force, so grip memory is not pure evidence of an updated object-dynamics model. | EMPIRICAL_FINDING | Action-history memory and separate grip/load influences | Missing | learning, prehension |
| FJ-029 | Object dynamics, pages 656-657 | Novel dynamics can initially be encoded in intrinsic or arm-centered coordinates; formation of one object-centered model remains open. | EMPIRICAL_FINDING and OPEN_QUESTION | Coordinate frame and unresolved consolidation relation | Missing | object_dynamics |
| FJ-030 | Object dynamics, page 656 | Transfer differs when the object is rotated versus when the hand orientation changes. | EMPIRICAL_FINDING | Transformation context and transfer result | Missing | object_dynamics, learning |
| FJ-031 | Object dynamics, page 657 | Sensorimotor weight control can adapt accurately while the size-weight perceptual illusion persists. | EMPIRICAL_FINDING | Separate sensorimotor and perceptual representations | Missing | object_dynamics, perception |
| FJ-032 | Conclusion, page 657 | Sensorimotor control points and the representation of object properties require further empirical and modeling elaboration. | OPEN_QUESTION | Explicit scope limitation and provenance | Missing | all scientific modules |

## Current core vocabulary classification

Disposition values describe the required direction, not an implementation action
for Phase 0.

| Current class | Scientific basis | Disposition | Contract note |
|---|---|---|---|
| CouplingRegime | ENGINEERING_EXTENSION compatible with the manipulation account | Revise | Coupling is independent of boundary and regulation dimensions. |
| ControlRegime | ENGINEERING_EXTENSION | Split | It currently makes discrete boundaries and continuous regulation exclusive. |
| ConditionMode | ENGINEERING_EXTENSION | Keep | A planning formalization for entry, invariant, and exit conditions. |
| ConditionPolarity | ENGINEERING_EXTENSION | Keep | A planning formalization not asserted by the paper. |
| ManipulationRole | ENGINEERING_EXTENSION compatible with the paper | Keep | Generic participant-role abstraction. |
| Manipulandum | SUPPORTED_INTERPRETATION | Keep | Represents the object whose state is manipulated. |
| Actor | EMPIRICAL_FINDING compatible | Keep | Must remain distinct from the effector role. |
| RoleSlot | ENGINEERING_EXTENSION | Keep | Resolution infrastructure, not a scientific claim. |
| MechanicalEventType | AUTHOR_HYPOTHESIS framework | Revise | Must support overlapping physical characteristics. |
| ContactEventType | EMPIRICAL_FINDING and AUTHOR_HYPOTHESIS framework | Revise | Contact change must not be disjoint from threshold crossing. |
| ForceDynamicEventType | ENGINEERING_EXTENSION grounded in force findings | Revise | Treat force threshold as an event characteristic. |
| ControlEventType | ENGINEERING_EXTENSION | Revise | Setpoint attainment is not necessarily a mechanical kind. |
| SetpointAttained | ENGINEERING_EXTENSION | Revise | Relate a controlled variable and task or motor setpoint explicitly. |
| PhaseConditionType | ENGINEERING_EXTENSION | Keep | Generic world-state description. |
| Subgoal | AUTHOR_HYPOTHESIS framework | Revise | Keep the goal distinct from evidence of attainment. |
| ConditionRequirement | ENGINEERING_EXTENSION | Keep | Formal phase-condition qualification. |
| MagnitudeRange | ENGINEERING_EXTENSION | Revise | Add dimension-safe scalar quantities; do not use for categorical values. |
| Parameter | ENGINEERING_EXTENSION | Split or retain as abstract root | Parameter categories must remain semantically distinct. |
| ObjectPropertyEstimate | EMPIRICAL_FINDING family | Split | It currently conflates property, interface, state, and dynamics estimates. |
| MotorSetpoint | SUPPORTED_INTERPRETATION | Keep | Must contain only motor-control targets. |
| ForceRateSetpoint | EMPIRICAL_FINDING | Keep | Required for anticipatory force-rate scaling. |
| MechanicalThreshold | ENGINEERING_EXTENSION grounded in events | Keep and revise | Threshold must carry a dimension and define the crossing event. |
| SafetyMargin | EMPIRICAL_FINDING | Keep | Must participate in the grip/slip relationship. |
| SensorySignalType | EMPIRICAL_FINDING | Split | Separate task-level sensory event, modality, feature, and afferent encoding. |
| TactileSignal | EMPIRICAL_FINDING | Revise | Represent modality or afferent feature without conflating prediction and observation. |
| ProprioceptiveSignal | EMPIRICAL_FINDING | Revise | Same separation as tactile signals. |
| VisualSignal | EMPIRICAL_FINDING | Revise | Visual cue, predicted visual event, and actual visual event are different. |
| InternalModel | SUPPORTED_INTERPRETATION | Move or revise | Generic predictive model belongs with prediction or object dynamics. |
| ForwardModel | SUPPORTED_INTERPRETATION | Move and connect | Must consume commands, efference copy, and represented dynamics and produce predictions. |
| InverseModel | SUPPORTED_INTERPRETATION extrapolated from command prediction | Move and connect | The review supports the function more clearly than this exact class taxonomy. |
| ObjectRepresentation | SUPPORTED_INTERPRETATION | Move and revise | Represented content must not be disjoint from models that use it. |
| CorrectiveActionPattern | EMPIRICAL_FINDING | Move and expand | Add trigger, target command, direction, effect, and timing. |
| SensorimotorControlPoint | AUTHOR_HYPOTHESIS | Move and expand | Must reify a genuine predicted-versus-actual comparison and its outcomes. |
| ActionPhase | AUTHOR_HYPOTHESIS framework | Keep and revise | A phase may have a discrete boundary and continuous regulation simultaneously. |
| PhaseTransition | ENGINEERING_EXTENSION | Keep and revise | Gate on verified control outcomes where required, not raw event occurrence alone. |
| ActionPhaseStructure | ENGINEERING_EXTENSION formalizing the paper | Keep | Root description for generic phase-control knowledge. |

## Current prehensile and pouring vocabulary classification

The current pouring module contains both generic prehension and genuine pouring
extensions. That functional debt must be removed in Phase 5.

| Current class | Scientific basis | Disposition | Contract note |
|---|---|---|---|
| GraspingEffector | EMPIRICAL_FINDING compatible | Move to prehension | The hand or gripper is an effector, not the actor. |
| GraspSite | EMPIRICAL_FINDING | Move to prehension | Digit placement and local contact geometry affect manipulation. |
| InitialSupport | EMPIRICAL_FINDING compatible | Move to prehension | Generic support role. |
| FinalSupport | EMPIRICAL_FINDING compatible | Move to prehension | Generic support role; may equal the initial support. |
| ContactEstablishment | EMPIRICAL_FINDING | Move to prehension and revise | Contact change characteristic rather than an exclusive event taxonomy branch. |
| ContactBreak | EMPIRICAL_FINDING | Move to prehension and rename if needed | Pair consistently with contact establishment or termination. |
| Liftoff | EMPIRICAL_FINDING | Move to prehension and correct | Simultaneously breaks support contact and crosses a force condition. |
| Touchdown | EMPIRICAL_FINDING | Move to prehension and correct | Simultaneously establishes support contact and changes load support. |
| SlipEvent | EMPIRICAL_FINDING | Move to prehension | Must relate load, friction, grip force, and contact state. |
| LiftClearanceReached | ENGINEERING_EXTENSION | Move to prehension | Optional task-specific boundary, not a universal Flanagan phase marker. |
| StableHoldReached | ENGINEERING_EXTENSION | Move to prehension | Explicit stable-hold boundary for the reference structure; optional rather than a universal Flanagan marker. |
| WithdrawalComplete | ENGINEERING_EXTENSION | Move to prehension | Explicit completion boundary for optional post-release withdrawal. |
| PourPoseReached | ENGINEERING_EXTENSION | Keep in pouring | Pouring-specific target-pose event. |
| EffectorObjectContact | EMPIRICAL_FINDING compatible | Move to prehension | Generic contact state. |
| ObjectSupportContact | EMPIRICAL_FINDING compatible | Move to prehension | Generic support-contact state. |
| GraspStability | EMPIRICAL_FINDING | Move to prehension | Must be grounded in grip force, load, friction, and safety margin. |
| ObjectClearOfSupport | EMPIRICAL_FINDING compatible | Move to prehension | Generic post-liftoff state. |
| EffectorPositioned | ENGINEERING_EXTENSION | Move to prehension | Generic reach precondition or subgoal. |
| ReachPhase | EMPIRICAL_FINDING compatible | Move to prehension | Contact is its likely boundary, but reaching is not always required. |
| LoadPhase | EMPIRICAL_FINDING | Move to prehension and rename | Force loading begins after contact and should not display as grasping. |
| LiftPhase | EMPIRICAL_FINDING compatible | Move to prehension | Separate post-liftoff lift only when the task needs it. |
| HoldPhase | ENGINEERING_EXTENSION compatible | Move to prehension and make optional | Not mandatory in every manipulation sequence. |
| TransportPhase | EMPIRICAL_FINDING | Move to prehension | Must permit continuous grip/load prediction with a discrete boundary. |
| ReplacePhase | EMPIRICAL_FINDING compatible | Move to prehension | Ends at support-contact establishment. |
| UnloadPhase | EMPIRICAL_FINDING compatible | Move to prehension and correct boundary | Ends when forces unload, not at contact break. |
| ReleasePhase | EMPIRICAL_FINDING compatible | Move to prehension and correct boundary | Contact break terminates release; withdrawal is a separate optional phase. |
| GripForceSetpoint | EMPIRICAL_FINDING | Move to prehension | Normal force target with an explicit role and dimension. |
| LoadForceSetpoint | EMPIRICAL_FINDING | Move to prehension | Tangential load-force target with an explicit role and dimension. |
| SlipThreshold | EMPIRICAL_FINDING | Move to prehension | Derived from load, friction, geometry, and force convention. |
| LiftoffThreshold | EMPIRICAL_FINDING | Move to prehension | Compare load force with weight force using consistent dimensions. |
| ExpectedWeight | EMPIRICAL_FINDING | Move, rename, and disambiguate | Use MassEstimate or WeightForceEstimate rather than kilograms under a weight name. |
| ExpectedFriction | EMPIRICAL_FINDING | Move and rename | A contact-interface estimate, not an object-only property. |
| ExpectedCenterOfMass | EMPIRICAL_FINDING | Move and rename | A spatial or mass-distribution estimate, not an untyped scalar. |
| ExpectedShape | EMPIRICAL_FINDING | Move and rename | Categorical or geometric representation, never a qualitative numeric range. |
| GripLoadCoupling | EMPIRICAL_FINDING | Move to prehension and expand | Include temporal coupling, predicted load, friction, and safety margin. |
| SourceContainer | ENGINEERING_EXTENSION | Keep in pouring | Pouring-specific participant role. |
| TargetReceptacle | ENGINEERING_EXTENSION | Keep in pouring | Pouring-specific destination role. |
| PouredSubstance | ENGINEERING_EXTENSION | Keep in pouring | Distinguish substance from manipulated container. |
| PouringLip | ENGINEERING_EXTENSION | Keep in pouring | Geometric flow locus. |
| FlowEventType | ENGINEERING_EXTENSION | Keep in pouring and revise | Flow change may coexist with threshold or contact characteristics. |
| FlowOnset | ENGINEERING_EXTENSION | Keep in pouring | Material-regime-specific flow-start event. |
| FlowCessation | ENGINEERING_EXTENSION | Keep in pouring | Material-regime-specific flow-stop event. |
| SpillEvent | ENGINEERING_EXTENSION | Keep in pouring | Failure event requiring geometry and destination relations. |
| TargetVolumeReached | ENGINEERING_EXTENSION | Keep in pouring and generalize | A task quantity may be mass, count, or volume depending on substance. |
| LipAligned | ENGINEERING_EXTENSION | Keep in pouring | Alignment setpoint event. |
| UprightRestored | ENGINEERING_EXTENSION | Keep in pouring | Optional post-flow event. |
| AlignmentPhase | ENGINEERING_EXTENSION | Keep in pouring and make optional | Not required when the source begins aligned. |
| TiltToFlowPhase | ENGINEERING_EXTENSION | Keep in pouring | One possible actuation strategy, primarily for pourable containers. |
| FlowRegulationPhase | ENGINEERING_EXTENSION | Keep in pouring | Must identify controlled variables, predicted trajectories, and observations. |
| FlowTerminationPhase | ENGINEERING_EXTENSION | Keep in pouring | Termination strategy depends on material and goal. |
| ReturnUprightPhase | ENGINEERING_EXTENSION | Keep in pouring and make optional | Not universal to all substance transfers. |
| ExpectedViscosity | ENGINEERING_EXTENSION | Keep in pouring and rename | Applies to appropriate fluid regimes, not granular cereal. |
| ExpectedFillLevel | ENGINEERING_EXTENSION | Keep in pouring and reclassify | A current source-state estimate rather than an inherent property. |
| ExpectedSloshDynamics | ENGINEERING_EXTENSION | Keep in pouring and reclassify | A liquid dynamics model, not a scalar object property. |
| ExpectedCenterOfMassShift | ENGINEERING_EXTENSION grounded in dynamics | Keep in pouring and reclassify | A predicted trajectory or changing state. |
| TiltAngleSetpoint | ENGINEERING_EXTENSION | Keep in pouring | Motor setpoint with angle dimension. |
| TiltRateSetpoint | ENGINEERING_EXTENSION | Keep in pouring | Motor setpoint with angular-rate dimension. |
| TorqueSetpoint | ENGINEERING_EXTENSION grounded in prehension | Place by ownership | Generic grasp torque may belong in prehension; pouring-specific use belongs in pouring instances. |
| TargetVolumeSetpoint | ENGINEERING_EXTENSION | Rename and reclassify | A task setpoint, not a motor setpoint. |
| FlowOnsetThreshold | ENGINEERING_EXTENSION | Keep in pouring and revise | Depends on container geometry, material state, and actuation. |
| SpillThreshold | ENGINEERING_EXTENSION | Keep in pouring and revise | Spilling is multivariate and not generally one scalar threshold. |
| AuditorySignal | EMPIRICAL_FINDING | Move to prediction or core sensory vocabulary | Auditory contact evidence is general, not pouring-specific. |
| ContinuousSignalPrediction | SUPPORTED_INTERPRETATION | Move to prediction and expand | Continuous prediction is general and must identify signal and trajectory. |
| PouringPhaseStructure | ENGINEERING_EXTENSION | Keep in pouring | Composition root for pouring-specific knowledge. |

## Competency questions

The finished microworld must answer these through typed structures and typed
query methods. Free-text labels alone do not count as an answer.

### Generic phase structure

| Question | Required answer |
|---|---|
| CQ-001 | What participant roles does this manipulation require? |
| CQ-002 | What phases are possible or required for this task variant? |
| CQ-003 | What subgoal does each phase pursue? |
| CQ-004 | What physical event or verified condition normally ends each phase? |
| CQ-005 | Which contacts are established or broken by a boundary event? |
| CQ-006 | Which phase may follow a successfully verified control point? |
| CQ-007 | Which phases are optional because the initial or desired state already satisfies their subgoal? |
| CQ-008 | Which conditions are entry requirements, invariants, exit outcomes, or world assumptions? |

### Prediction and comparison

| Question | Required answer |
|---|---|
| CQ-009 | What sensory events are predicted at a control point? |
| CQ-010 | In which modalities, locations, participants, and timing windows are they predicted? |
| CQ-011 | What actual observation type is compared with each prediction? |
| CQ-012 | Did the event match in occurrence, timing, location, magnitude, and direction? |
| CQ-013 | What prediction-error kind describes a mismatch? |
| CQ-014 | What immediate corrective action follows that error? |
| CQ-015 | Which motor setpoint or action phase does the correction modify? |
| CQ-016 | Does the controller repeat, continue, terminate, or replan the phase? |
| CQ-017 | What representation or estimate is updated for future actions? |
| CQ-018 | What initial-state estimate is supplied to the next phase? |
| CQ-019 | What continuous sensory trajectory is predicted within a phase? |
| CQ-020 | What signal, tolerance, controlled variable, and model define that continuous comparison? |

### Internal models and learning

| Question | Required answer |
|---|---|
| CQ-021 | Which cues and memories support an object-property or dynamics estimate? |
| CQ-022 | Which motor setpoints are scaled by each estimate? |
| CQ-023 | How do an efference copy and dynamics model produce a predicted load? |
| CQ-024 | Is a load represented as acting on the object, the limb, or both? |
| CQ-025 | Which coordinate frame does a representation use? |
| CQ-026 | Is the representation used for sensorimotor control or perceptual judgment? |
| CQ-027 | Can perceptual and sensorimotor estimates disagree without invalidating either representation? |
| CQ-028 | Is the model active only while a particular object is grasped? |
| CQ-029 | Is learning discovering a new mapping or adapting an existing one? |
| CQ-030 | What trial, cue condition, or error controls the update rate? |

### Prehensile manipulation

| Question | Required answer |
|---|---|
| CQ-031 | What grip force, load force, friction estimate, and safety margin define grasp stability? |
| CQ-032 | How are grip force and predicted load coupled in time? |
| CQ-033 | What physical and contact conditions constitute lift-off? |
| CQ-034 | What physical and contact conditions constitute touchdown? |
| CQ-035 | What signals indicate slip and what protective response follows? |
| CQ-036 | What happens when expected lift-off is absent because the object is heavier than expected? |
| CQ-037 | What happens when lift-off occurs earlier because the object is lighter than expected? |
| CQ-038 | How does grip control anticipate an externally timed impact? |
| CQ-039 | Does a phase remain continuously regulated even though it has a discrete boundary? |

### Gaze and observation

| Question | Required answer |
|---|---|
| CQ-040 | What forthcoming contact location is the gaze target for a phase? |
| CQ-041 | How far ahead of the hand or held object should gaze arrive? |
| CQ-042 | Until which subgoal or event should fixation be maintained? |
| CQ-043 | Which visual event is aligned with tactile, proprioceptive, or auditory evidence? |
| CQ-044 | What contact site does an observer predict while watching an actor? |
| CQ-045 | Which modalities are available to the actor and to the observer? |
| CQ-046 | How does actor-hand visibility change observer gaze from proactive to reactive? |
| CQ-047 | Is actor-observer matching an empirical finding or an author hypothesis? |

### Learning and perception

| Question | Required answer |
|---|---|
| CQ-048 | Is the learner in exploration, skill acquisition, or skill refinement? |
| CQ-049 | Is gaze chasing the controlled object or proactively targeting the goal? |
| CQ-050 | What mapping is being discovered or adapted? |
| CQ-051 | Is a sensory event predicted to be self-generated or externally generated? |
| CQ-052 | What temporal window defines perceptual attenuation around predicted contact? |
| CQ-053 | Is the relationship between attenuation and action-control prediction established or unresolved? |

### Pouring extension

| Question | Required answer |
|---|---|
| CQ-054 | Which generic prehension phases are required by this particular pouring initial state? |
| CQ-055 | Is the transferred material liquid, granular, cohesive, or otherwise modeled? |
| CQ-056 | What conditions produce flow onset and flow cessation for that material regime? |
| CQ-057 | What flow, quantity, pose, and alignment variables are continuously controlled? |
| CQ-058 | What sensory trajectories provide feedback during transfer? |
| CQ-059 | Is transfer terminated by target quantity, empty source, explicit instruction, or failure? |
| CQ-060 | How does changing content alter center of mass, torque, grip force, and load prediction? |
| CQ-061 | What constitutes a spill and what correction or recovery follows? |

### Provenance and applicability

| Question | Required answer |
|---|---|
| CQ-062 | Which source claim and evidence status support this answer? |
| CQ-063 | Is the answer generic knowledge, a world assumption, an actual observation, or an engineering extension? |
| CQ-064 | Which task capability caused the defining module to be selected? |
| CQ-065 | Is this hypothesis valid for the current material, participant binding, and initial state? |

## Reference scenarios required by later tests

Reference scenarios are independent behavior specifications. They must not be
embedded as JSON or code strings inside test methods.

| Scenario | Claims exercised |
|---|---|
| Normal precision lift | FJ-003 through FJ-012, FJ-025 |
| Heavier-than-expected lift | FJ-005, FJ-007, FJ-010, FJ-011 |
| Lighter-than-expected lift | FJ-010 and FJ-011 |
| Unexpected low friction and slip | FJ-009, FJ-012, FJ-025 |
| Transport with a discrete destination and continuous grip-load control | FJ-003 and FJ-008 |
| Predictable ball impact with a catch trial | FJ-008 and FJ-025 |
| Object load versus arm load | FJ-026 and FJ-027 |
| Object rotation versus hand rotation | FJ-029 and FJ-030 |
| Size-weight illusion | FJ-031 |
| Sequential gaze during grasp, move, and replace | FJ-014 and FJ-015 |
| Observer with visible and hidden actor hand | FJ-016 through FJ-018 |
| Novel visuomotor mapping across learning stages | FJ-019 through FJ-022 |
| Predicted self-generated tapping contact | FJ-023 and FJ-024 |
| Liquid pouring | Engineering extension using generic predictive control |
| Granular cereal pouring | Engineering extension requiring non-viscous material dynamics |

## Constraint requirements derived from the contract

Later acceptance constraints must make these contradictions inadmissible:

- A contact-changing event classified as disjoint from contact change
- A phase forced to choose between a discrete boundary and continuous regulation
- A predicted sensory event with no modality, predicted feature, or generating
  model
- A comparison with no independently represented prediction and observation
- A normal transition that bypasses a required comparison result
- A corrective action with no triggering error or affected control variable
- A model update with no updated estimate or representation
- A continuous controller with no setpoint, controlled variable, monitored
  signal, or predicted trajectory
- A property estimate with no represented participant or source representation
- A quantity whose unit does not match its physical dimension
- A mass range used as a weight-force threshold without conversion
- A categorical property encoded as a fabricated numerical interval
- A motor setpoint used for a task-level desired quantity
- A perceptual estimate silently reused as a sensorimotor estimate
- A pouring-only concept emitted for a plain grasping task
- An engineering extension presented as an empirical finding

## Phase 0 exit review

Phase 0 is complete when all statements below are true:

- The scientific authority and epistemic statuses are explicit.
- Every major section of the review maps to at least one claim identifier.
- Every current core and pouring class appears in the classification tables.
- Every current class has a documented keep, revise, move, split, rename, or
  reclassify disposition.
- The non-negotiable semantic distinctions cover every contradiction identified
  in the initial audit.
- Functional ownership and one-way module dependencies are explicit.
- Competency questions cover generic phase control, prediction, object dynamics,
  prehension, gaze, observation, learning, perception, pouring, and provenance.
- Later constraints have a documented list of contradictions to reject.
- No production class, registry entry, query API, or acceptance constraint was
  changed during Phase 0.

## Phase 1 handoff

Phase 1 begins by adding a whole-structure validation test for the existing
hand-authored cereal-pouring structure. The test must expose the current
acceptance errors before fixes are made and must ultimately require zero errors
and zero unexplained warnings.

The first Phase 1 fixes are structural validity, not ontology redesign:

- Bind every parameter to its participant roles.
- Give every continuously regulated phase a monitored signal.
- Ground or explicitly mark all assumptions.
- Replace generic boundary instances where the current classifier requires a
  concrete type.
- Give each current control point an explicit and justified mismatch policy.

Foundational event, control, parameter, and quantity redesign begins only in
Phase 2 after Phase 1 establishes a validated reference baseline.
