# Answering competency questions about an action: design note

Agreed with the user on 2026-10-25; nothing is built yet. The knowledge component is to
answer questions about an action at every level of competence — from *what is pouring*
to *which constraints keep this pour safe* — besides serving the resolver, and each
answer says what it rests on. A language model gives the generic understanding of the
action, as a hypothesis; physics confirms or refutes every part of it that physics results
reach. How a question is written (an EQL query of some shape) is deferred; this note fixes
what is asked, what answers it and what backs the answer. It builds on
[`architecture.md`](architecture.md) (§2.3 the knowledge component, §2.6 physical
measurements) and on the competency questions CQ-001 to CQ-065 of
[`flanagan_scientific_contract.md`](flanagan_scientific_contract.md), which already
require typed answers rather than free text.

## 1. Decisions

| Decision | Consequence |
|---|---|
| A language model hypothesises, physics confirms | The model's pretrained understanding of actions — procedures, roles, phases, events and their order, forces, constraints, failures — is the source of the generic knowledge, always as a hypothesis (Generated, §3). Every part that a physics result reaches is confirmed or refuted by it: a quantity derived or measured, an event observed to occur, an order observed in time, a limit found by a sweep. What no physics result reaches stays Generated, and says so |
| `llm_reasoner` stays independent of any world model | It imports krrood (and `random_events`, `probabilistic_model` through it) and never `semantic_digital_twin`, `coraplex`, `cramera` or a simulator. Physics results reach it as records in its own ontology terms (§4); the world reaches it only through `WorldAccess` over krrood's symbol graph |
| Other packages are used as they are | krrood, `semantic_digital_twin`, `coraplex`, `giskardpy`, `physics_simulators` and the others are read and called, never changed; what they lack is built on our side or asked about. segmind is not used |
| The knowledge has two consumers: the resolver and a question answerer | Questions are read through the same knowledge graph, measurement store and world access as resolutions; neither consumer changes the other |
| An answer is typed: the fragments, measurements and world individuals it consists of, each cited by its IRI or handle | A language model may write the answer in prose afterwards, citing what it rests on; it adds no fact |
| Every part of an answer carries an evidence grade (§3) | An answer never presents a generated value as measured; a part physics contradicts is reported as refuted, not left out |
| Answers are read from reviewed, frozen knowledge (§6) | The same question gets the same answer; the 21 generations of the same liquid pouring request held 7 different flow-onset ranges (`status.md`, 2026-10-25) |
| Questions are written as EQL queries; their exact form is deferred | The catalogue of §5 states each question's meaning and answer shape, so the form can be chosen later |
| The work stays in this repository | The separate literature project (MAKF) is set aside (user, 2026-10-25) |

## 2. Levels of competence

The levels follow Bloom's taxonomy, revised: each asks more of the knowledge than the one
before, and the higher ones need the world as well.

| Level | What it asks | Example questions (pouring) | Competency questions | What answers it | What can back it |
|---|---|---|---|---|---|
| Remember | What the action and its parts are | What is the action? Which roles does it involve? What material regimes can be poured? | CQ-001, CQ-055 | The ontology, read by EQL over its classes; the action core's frame | Defined; the frame's own content Generated |
| Understand | How the action is organised | What are the phases, in which order? What does each phase pursue, and which event ends it? Which conditions hold throughout? | CQ-002–CQ-004, CQ-006–CQ-008, CQ-059 | A reviewed phase structure and its perspectives (`HypothesisView.phases`, `next_phases`, `requirements_of`, `world_assumptions`) | Generated, Sourced where a paper claim says it; the events and their order Measured once episodes are observed (§4) |
| Apply | The knowledge carried to one situation | Which phases does this initial state require? At which tilt does this pitcher, half full, start to pour? Is this hypothesis valid for this material and these participants? | CQ-054, CQ-056, CQ-065 | The resolver's world access with the knowledge and the measurement store | Measured or Derived |
| Analyze | The forces and events inside the action | Which force-dynamic events occur, and what opposes what in each? Which contacts does a boundary event make or break? How does the content shift the centre of mass and the grip needed? | CQ-005, CQ-009–CQ-011, CQ-031–CQ-035, CQ-060 | `HypothesisView.force_dynamics`, `force_couplings_in`; events observed in simulated episodes | Measured where an episode shows the event, else Sourced or Generated |
| Evaluate | Whether an action meets its constraints | Which constraints must hold for a successful pour? Does this designator break one? Did the pour's events match their predictions? What counts as a spill, and what corrects it? | CQ-012–CQ-016, CQ-061 | Constraints checked against measured limits and against a trial's outcome | Measured |
| Create | A new executable action | An executable designator for this situation; the phases a new initial state needs | — | The resolver, the outcome check and, later, planning | All of the above, cited |

The questions on provenance (CQ-062, CQ-063) are not a level of their own: every answer
answers them, through its evidence grades.

## 3. Evidence grades

| Grade | Meaning | Where it lives today |
|---|---|---|
| Measured | Observed or measured in a physics simulation, with its setup: a quantity, an event's occurrence, the order of events, a limit | `PhysicsSimulationSetup` records (`physical_measurements/flow-onset.ttl`) |
| Derived | Computed from geometry or another physical model, with its setup | `GeometricDerivationSetup` records |
| Sourced | Stated by a source, with the source's own support | `core:SourceClaim`, `core:claimedIn` and `core:evidenceStatus` (`EmpiricalFinding`, `SupportedInterpretation`, `AuthorHypothesis`, `OpenQuestion`) in the predictive manipulation core |
| Defined | Follows from the ontology's axioms: a class, a restriction, a CRAM-AICOR or DUL term | The ontology modules |
| Generated | Stated by a language model and admitted by the acceptance checks; a hypothesis | Hypothesis fragments; the record's `client_identity` names the model |

- `core:evidenceStatus` grades a class against the literature; the comment of the property
  says it is "not confidence in a generated individual". The grades above are about one
  value or fragment of an answer, so Sourced reuses `evidenceStatus` and the other grades
  are new.
- A part of an answer is graded by the strongest evidence that bears on it, and an answer
  states the grade of each part; a summary grade, where one is shown, is the weakest part's.
- Evidence can contradict a hypothesis: a measured flow onset of 55.2° *supports* a
  generated onset range of 30–110° and *refutes* one of 70–110°; an observed lift-off before
  the grasp closes would refute a structure that orders them the other way. An answer
  shows the hypothesis and the verdict together (§7, step C1).
- Ontology classes that are an `EngineeringExtension` (every class of `pouring.ttl`) back a
  Defined answer about what the vocabulary says, not a Sourced one about the world.

## 4. Physics evidence for events and constraints

### 4.1 Observations in the ontology's own terms

The ontology already defines each event by the state changes it makes. For example,
`ContactEventType` is "a mechanical event that makes or breaks contact", and `Liftoff` (in
`prehension.ttl`):
- terminates `ObjectSupportContact` and initiates `ObjectClearOfSupport`;
- resolves the load force against the weight;
- ends the support reaction.

A physics episode is therefore recorded as observed changes in the same terms, never in
the world's own names. Each observed change says:
- which state type begins or ends (a contact condition, a flow state, an opposition between
  force kinds) between which roles' fillers;
- when it happens.

`llm_reasoner` classifies the observed changes into event types with the definitions that
already check generated hypotheses (closed-world membership, `acceptance/restrictions.py`).
It then compares two things with the hypothesis:
- which predicted events occurred;
- whether they occurred in the predicted order and within any predicted windows.

| Part | Where | Depends on |
|---|---|---|
| Vocabulary of observed changes and episodes | A module of the physical measurement family | The predictive manipulation core and action core |
| Recording an episode from a simulation: binding each body to the role it fills (the world side knows which body is the source, the target, the support), and writing the observed changes | Outside `llm_reasoner`: the physics harnesses in `experiments`, and cramera's trials | `physics_simulators`, `semantic_digital_twin` and coraplex, used as they are |
| Classifying observed changes into events, comparing them with a hypothesis, writing verdicts | `llm_reasoner` | krrood and its own modules only |

### 4.2 What a simulation can back

| Evidence | Observed or measured from | State |
|---|---|---|
| Flow onset of a liquid per container, fill and side of tilt | Isaac Sim PBD particles (`experiments/flow_onset_measurement`); the basin of the mesh | Built (`architecture.md` §2.6) |
| Contacts made and broken between role fillers: grasp, lift-off, touchdown, release | The simulator's contacts in a performed action (MuJoCo or Isaac Sim) | To build |
| Flow onset and cessation, spill, the transferred amount | Particles leaving the source, reaching the target, landing outside it | To build, extending the particle harness |
| Grip and load forces, slip | Contact forces and the held object's slip relative to the gripper | To build |
| Safety limits, such as the tilt rate at which a pour spills, the tilt at which the source tips, the grip at which it slips | Sweeps of the above over the parameter in question, each kept as a measurement | To build |

An observed event confirms that the hypothesis's event occurs, and in which order. A
measured limit backs a constraint. Neither turns a definition into a measurement: *what
pouring is* stays Defined. Where an observation needs a reading the simulator packages do
not expose publicly, for example contact and actuator forces through
`semantic_digital_twin`'s MuJoCo adapter (an open decision of the deferred proposal to
record physics runs as NEEMs), the harness reads what is exposed or the question is
taken to the user. Those packages are not changed.

## 5. The question catalogue

The answerer's questions are data, one entry each:

| Field | Content |
|---|---|
| Identifier | The contract's CQ number, or a new one for a question it lacks (such as the safety constraints) |
| Level | One of §2 |
| Meaning | The question in one sentence |
| Query | The EQL query over the knowledge graph (`KnowledgeGraph.instances_of` as its domains), the measurement store and, for Apply and above, the world through `WorldAccess` (form deferred) |
| Answer shape | The typed parts the answer consists of |
| Expected answer | For the test catalogue: the answer the reviewed knowledge and the measurements give |

Many entries need no new reading code:
- `HypothesisView` already answers phases, next phases, requirements, world assumptions,
  gaze fixations, force couplings, failure events, corrections and force dynamics;
- `MeasurementStore.flow_onset` answers flow onset.

Remember and Understand need no language model at the time of the answer; the model's
part was the hypothesis, generated before.

## 6. Reviewed, frozen knowledge

Generation stays the way to obtain knowledge for a new action or situation, but answers
are read only from knowledge a person has reviewed and frozen:

- A frozen record answers its key (root class, fixed values, module versions) in place of
  any other record of the same key, in the resolver and the answerer alike.
- Freezing keeps the record's provenance and adds who reviewed it and when.
- A frozen record's parts stay Generated until a physics result supports or refutes them.

## 7. Build order

Each step is test-first, keeps the existing suites passing, and is recorded in `status.md`.

| Step | What | Done when |
|---|---|---|
| C1 | Verdicts. A measurement *supports* or *refutes* a fragment that states the same quantity for the same situation, for example a measured onset inside or outside a generated `FlowOnsetThreshold` range. Verdicts are stored with the measurement family | Each of the six flow-onset records has a verdict on the stored liquid pouring hypotheses' onsets |
| C2 | Evidence grades on answers: a grade per part, from the record's provenance, `evidenceStatus`, the ontology and the verdicts | A Remember answer and an Understand answer come out graded on the shipped knowledge |
| C3 | Frozen knowledge: freezing a reviewed record, and the store serving frozen records first | The liquid pouring structure, frame and perspectives are frozen; resolver tests unchanged |
| C4 | The catalogue and the answerer for Remember and Understand, reusing `HypothesisView`; the expected answers as tests | The CQs of those levels are answered on the frozen knowledge |
| C5 | Observed episodes:<br>• the vocabulary of observed changes (§4.1);<br>• a recorder in the pouring trial and the harnesses, binding bodies to roles and writing the changes from the simulator's contacts and particles;<br>• in `llm_reasoner`, the classification into events and the comparison with a hypothesis's predicted events and order | A simulated pour's grasp, lift-off, flow onset, touchdown and release, observed in order, support or refute the frozen phase structure's events and order |
| C6 | Analyze and Evaluate: force-dynamic events and constraints, checked against measured limits and trial outcomes; spill and transferred amount from the particle harness | "Which constraints must hold?" and "does this designator break one?" are answered with Measured parts |
| C7 | Apply and Create through the resolver, its answers graded the same way | A resolution's grounds carry evidence grades |

## 8. Open questions

1. The form of a question query (deferred by the user).
2. Who reviews knowledge before it is frozen, and how a review is recorded.
3. How close a measured value must lie to a generated range for *supports* rather than
   *refutes*, and whether a range far wider than the measured spread is reported as vague.
4. Which simulator observes rigid-body contacts and forces (MuJoCo, Isaac Sim, or both),
   and how they are read without changing `semantic_digital_twin` or
   `physics_simulators`. The deferred NEEM proposal left four decisions open:
   - the storage format;
   - a public read interface for contacts and actuator forces;
   - KnowRob export;
   - sample rates and images.
5. Which action follows pouring, to show the catalogue is generic.
