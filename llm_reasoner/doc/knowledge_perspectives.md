# Knowledge perspectives: design note

Status (2026-10-19): **phases 0–4 are built and measured; phases 5 and 6 are built, tested and measured (§13.6); phases 7–8 are not started.**
The action core, the action frame, knowledge in both vocabularies, the roles-to-slots
mapping and the phase 4 comparison are done (§9, §11–§13); the action frame stays opt-in
(§13.4). It refines [`architecture.md`](architecture.md) §2.3 and answers the item
*Representing knowledge for tools* in [`status.md`](status.md) §6. Sections 2 and 5 are
written as the design stood on 2026-10-15; §9 and §13 record what was done since.

## 1. Why the knowledge component exists

The knowledge component is CRAM's Knowledge Executive, the part the plan asks when it is
*resolving parameters* (Beetz and Kümpel, *From Frames to Pouring: The CRAM Cognitive
Architecture for Everyday Robot Manipulation*, KI 2026, Fig. 2), with its content written
by a language model. The CRAM tradition describes the same action from several
perspectives, and each says something different about it:

| Perspective | Tradition | What it says about pouring |
|---|---|---|
| Action frame | FrameNet (Fillmore), frames (Minsky) | The roles (source, substance, goal, agent), what each is normally like, defaults, preconditions, what the action is for and what it brings about |
| Image schemas | Johnson, Lakoff, Talmy | The spatial and force structure underneath: the substance goes along a path from the source to the goal and into it; the source tips; the goal holds |
| Script | Schank and Abelson | The expected sequence of events, including preparation |
| Predictive control | Johansson and Flanagan | Phases, events, sensory predictions, prediction errors and corrections |

A language model knows all of this well, so it generates the representations. An
ontology keeps them consistent and queryable. The resolver reads them to understand an
underspecified designator as a person would (what the action is for, which object plays
which role, what is normally expected, what can go wrong), chooses the parameters, and
says why, citing the knowledge and the world.

| Component | Knows | Kind of truth |
|---|---|---|
| World injector | What *is* in this world now | Fact |
| Knowledge component | What an action *normally* involves, and why | Stereotype |
| Resolver | What to do *here* | A decision with its reasons |

## 2. Where we are

Only the predictive-control perspective exists (`knowledge/ontologies/predictive_manipulation/`).
It is the lowest-level one, and in a real run it said little the designator could use.

The run `baeca81c` (2026-10-14, `simple_pouring`) resolved
`the(PouringAction)(source_container=…, target_container=…, arm=..., tilt_angle=...,
pour_side=..., pour_side_offset=..., pour_approach_offset=..., pour_height=...)`:

- The hypothesis (12 phases, 23 setpoints, 9 requirements) was mostly about grip force,
  load rate and gaze, which the designator does not expose.
- The resolver read `knowledge_overview` once per attempt and cited one fragment, the
  tilt range of 30–120°, to confirm the default tilt of 1.85 rad.
- The first pour missed the opening by 0.188 m. The fix (`pour_side_offset = -0.188`)
  came from the simulation's outcome, not from the knowledge.
- The hypothesis did hold the condition that failed (`LipOverReceptacle`, on the lip and
  the target), but nothing linked it to the slot that positions the lip, and its only
  correction (`SpillCorrection`) acts on the tilt.

What the resolver lacked was frame and image-schema knowledge: the liquid must end in
the target; the lip must be over the target's opening; the target is an open container
with an opening of some size. With that, it would have known that the lateral offset
matters, why, and which world quantities decide it.

A limit to keep in mind: stereotypical knowledge does not produce numbers like
−0.188 m. It names what matters and how it constrains a parameter; the world supplies
the values, and experience (NEEMs) what worked before. The paper keeps semantic digital
twins and episodic memories as their own pillars for this reason.

## 3. Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Each perspective is its own ontology module, importing CRAM-AICOR (and DUL through it); CRAM-AICOR itself is not edited | Keeps CRAM-AICOR shared and stable; each module is small enough for a model to generate against and for a person to review |
| D2 | The reference CRAM-AICOR file is `cram-aicor-ontology.ttl` | It is a strict superset of `aicor-ontology.ttl` (same ontology IRI `http://aicor.knowledge/l2-ontology.owl`; 1,069 classes and 811 object properties against 760 and 578; every property of the smaller file is in it). The extra terms include the participant roles (`AgentRole`, …), `Affordance` and the CRAM designator and architecture terms |
| D3 | The model proposes the vocabulary of each perspective | The user's preference; each proposal is checked and reviewed before it is used (§6) |
| D4 | No SOMA. FrameNet and image-schema resources are used as seeds for the model, not imported | Framester (FrameNet as OWL, aligned with DOLCE-Zero) and ImageSchemaNet (image schemas as a Framester layer, each schema a frame with roles) are large hubs; their terms are cited with `rdfs:seeAlso` for provenance |
| D5 | Everything is stated in CRAM-AICOR terms where CRAM-AICOR has them | So the knowledge fits the CRAM architecture |
| D6 | The action core's frame of a manipulation sits under `aicor:ManipulationFrame` (user, 2026-10-15) | CRAM-AICOR's frame layer is where frame elements (`RoleSlot`s) already hang |
| D7 | The model proposes the vocabulary of one perspective at a time (user, 2026-10-15) | Each proposal stays small enough to review and understand before the next |
| D8 | After pouring: navigating, picking up and placing (user, 2026-10-15) | The actions of the coraplex designators `NavigateAction`, `PickUpAction` and `PlaceAction`, which a pouring plan also performs |
| D9 | A proposed vocabulary is reviewed by a person before it is used; a designator's roles-to-slots mapping is accepted when it passes automatic checks and is written to the mapping file in the repository, where a change shows as a diff (user, 2026-10-15) | The vocabulary is what everything else is built on and changes rarely; the mappings are small, one per designator, and a wrong one shows in the resolver's grounds |

## 4. What CRAM-AICOR already offers

CRAM-AICOR already shapes pouring and has a frame layer. The classes carry no
definitions, so the modules below give the meanings the generation prompt needs.

| Term | What it holds |
|---|---|
| `Pouring` ⊑ `Transferring` ⊑ `SubstanceManipulation` | Subactions `Tilting`, `Grasping`; `hasIntentionalStructure → PouringIntention`; `hasCausalStructure → PouringTemplate` |
| `PouringIntention` ⊑ `IntentionalStructure` | Participant roles `AgentRole`, `SourceRole`, `SubstanceRole`, `GoalRole`; `hasTelos → PouringTelos` |
| `PouringTelos` ⊑ `Telos` | `definedOverState → AccumulatedState` (the substance has accumulated in the goal) |
| `PouringTemplate` ⊑ `ForceDynamicTemplate` | `EffluxOnsetEvent`, `FlowingState`, terminal `AccumulatedState`, `PouringPhase` |
| `ManipulationFrame` | `frameOf → ManipulationProblemClass`, `hasFrameElement → RoleSlot` |
| `RoleSlot` ⊑ `dul:Concept`; `RoleBinding` | `physicallyTypedBy → MaterialProperty`; a binding links a `RoleSlot` to a `dul:PhysicalObject` filler, which is what the resolver's grounding is |
| `ParticipantRole` | `AgentRole`, `PatientRole`, `SourceRole`, `GoalRole`, `InstrumentRole`, `SubstanceRole` |
| `Goal` | `AchievementGoal`, `MaintenanceGoal`, `PreventionGoal`, scoped to a phase |
| `Container` (`contains → Substance`), `Vessel`, `Bowl`, `Cup`; `OpenContainerState`, `ContainedState`, `FlowingState` | The kinds of things and states pouring involves |
| `Effect`, `TaskRelativeEffect`, `Expectation`, `Affordance`, `ActionParameterSpace` | Consequences, expectations, what something affords, and parameters |
| `ForceDynamicSchema`, `QualiaStructure` | Talmy-style force dynamics with role slots; telic, constitutive and agentive roles |

### 4.1 The code's own schema

Since 2026-10-15 the schema used here is `knowledge/ontologies/cram_classes/cram-classes.ttl`,
written by `llm_reasoner.world.class_ontology` from the selection beside it: CRAM's action
and motion designators, semantic annotations and robots, with coraplex's data structures in
full and the designators' bookkeeping left out (392 classes, 13 enumerations). It keeps what
is useful below and none of what is not wanted. What follows was found in the hand-made
`cram_tbox.ttl` it replaces.

`~/ontologies/cram_tbox.ttl` is generated from the Python source of the workspace
(coraplex, giskardpy, krrood, probabilistic_model, random_events, robokudo,
semantic_digital_twin): 3,122 classes, 2,154 object and 2,128 datatype properties; a class
is a Python class with its docstring, a property a field scoped to its class, and
inheritance is Python inheritance. Its own description says it is not a curated taxonomy.

Useful here:

- **The designator side of the roles-to-slots mapping.** Each action designator with its
  fields, their types and docstrings, such as `PouringAction` (`source_container: Tool`,
  `target_container: Body`, `tilt_angle: double` in radians, `pour_side_offset: double`
  in metres, …), `NavigateAction` (`target_location: Pose`, `keep_joint_states`),
  `PickUpAction` (`object_designator: HasRootBody`, `arm`, `grasp_description`) and
  `PlaceAction` (`object_designator: Body`, `target_location: Pose`, `arm`). As RDF it can
  be joined with CRAM-AICOR and the perspectives, and a mapping can be kept as triples
  linking a field's IRI to a role or a parameter dimension.
- **The world's kinds.** The semantic annotations (124 classes) are the kinds a role's
  filler will be.

Not wanted, and to be left out by a filter rather than by hand: exceptions (295 classes),
classes from test paths (44), ROS middleware, adapters and visualisers, krrood's own
internals (the query verbaliser alone has 264 classes), private fields such as
`_action_plan`, implementation mixins as superclasses (`HasTcpGoalThresholds`,
`PickUpTuningParameters`), and the placeholder namespace `https://example.org/cram/tbox`.
It does not link to CRAM-AICOR. The slice worth keeping is the coraplex actions, their
data structures and enums, and the semantic annotations.

**What it shows about kinds.** Neither the code nor CRAM-AICOR says that a bowl holds
things. In the code, `Bowl` is a `HasSupportingSurface` and `IsPerceivable`, `Cup` a
`DrinkingContainer`, `PouringCup` a `Tool`, and `Milk` (documented as "a container of
milk") a `Food`. In CRAM-AICOR, `Bowl` and `Cup` are `Vessel`s, and `Vessel` is a
`Representation` (an information object), while `Container` is a separate physical class.
So a role's expectations are stated as what the filler must have or afford (an opening,
the ability to hold a substance, an upright stance), which a world can check, and not as
membership of a class such as `Container`, which neither the code's nor CRAM-AICOR's
classification would confirm.

## 5. The modules

```
CRAM-AICOR (cram-aicor-ontology.ttl, imports nothing; DUL terms as dul_*)
   └── action-core          the hub: an action's roles, telos, effects, parameter dimensions
         ├── action-frame   per role: expectations, typical fillers, defaults; preconditions, consequences
         ├── image-schemas  spatial and force relations between roles
         ├── failures       failure modes and corrections, linked to roles and parameter dimensions
         └── predictive-control   the existing Flanagan modules, re-anchored to the core's roles
```

Each module imports the action core, and through it CRAM-AICOR. A perspective never
refers to another perspective, only to the core, so perspectives can be generated,
reviewed and offered independently, and a hypothesis can hold any subset of them.

### 5.1 The action core

For each action concept (`aicor:Pouring`) the core states:

| Content | In CRAM-AICOR terms (proposed) |
|---|---|
| The action and its frame | An `ActionFrame` ⊑ `aicor:ManipulationFrame` (D6), linked to the action concept; for an action that is not a manipulation, such as navigating, see §10 |
| Its roles | `aicor:RoleSlot`s as frame elements (`hasFrameElement`), each classified by an `aicor:ParticipantRole` (source, substance, goal, agent, instrument), and parts of roles where they matter (the source's lip, the goal's opening) |
| What it is for | The concept's `aicor:Telos` and `aicor:Goal`s (for pouring: `PouringTelos`, the substance accumulated in the goal) |
| What it brings about | `aicor:Effect`s on the roles: the goal holds more, the source less |
| Its parameter dimensions | Kinds of parameter, independent of any designator: amount, rate, where relative to the goal, orientation of the source, height above the goal; each a specialisation of `aicor:ActionParameterSpace` with a physical dimension and unit |

The roles and the parameter dimensions are what the perspectives refer to, and what the
resolver maps to a designator's slots.

### 5.2 The action-frame perspective (first)

For each role of the core:

| Content | Pouring example |
|---|---|
| Expected kind of filler | Source: a `Container` or `Vessel` holding a `Substance`. Goal: a `Container` with an opening |
| Expectations on the filler, as conditions on properties | Source: in `OpenContainerState`, `contains` the substance, upright before pouring. Goal: in `OpenContainerState`, has room for the amount |
| Defaults | Substance: a liquid. Amount: until the goal holds what is wanted, not until the source is empty |
| Affordances | The source affords tipping (it has a lip); the goal affords containing |

For the action as a whole: preconditions (the agent holds the source; the source's lip
can be brought above the goal's opening), consequences, and the role-level reason behind
each parameter dimension.

Expectations are stated as conditions a world can check, in ontology terms
(`OpenContainerState`, `contains`), never in terms of a particular world, and as what the
filler has or affords rather than which class it belongs to (§4.1). The resolver
grounds a role in an individual through `RoleBinding`, then checks the expectations on
that individual with its world tools.

FrameNet seeds the model with the frame elements of the frames `pour` evokes; the
module's roles cite them with `rdfs:seeAlso`.

### 5.3 The image-schema perspective (second)

Relations between the core's roles, using the schemas pouring instantiates:

| Schema | Pouring |
|---|---|
| SOURCE–PATH–GOAL | The substance goes from the source's lip along a path to the goal's opening |
| CONTAINMENT | The substance is in the source before and in the goal after; the goal's opening is the way in |
| VERTICALITY | The source tips from upright; the lip is above the opening |
| SUPPORT, CONTACT | The goal stands on a support; the agent's effector holds the source |

Each relation names the parameter dimensions it constrains ("the lip over the opening"
constrains *where relative to the goal* and *height above the goal*) and the world
quantities that decide it (the opening's centre and size, the lip's position). This is
the knowledge whose absence cost the real run its first pour. ImageSchemaNet seeds the
schema names and their roles. CRAM-AICOR's `ForceDynamicSchema` and `PouringTemplate`
belong here or beside the predictive-control perspective.

### 5.4 Failures and corrections (third)

Failure modes linked to the roles, relations and parameter dimensions they concern, with
corrections stated as changes to those dimensions: "the substance lands beside the goal"
means the lip was not over the opening, so change where relative to the goal. Today's
`SpillCorrection` acts only on the tilt.

### 5.5 Predictive control (kept)

The Flanagan modules stay as they are, with their roles (`SourceContainer`, `PouringLip`,
`TargetReceptacle`) declared as the core's roles, so all perspectives talk about the same
participants.

### 5.6 Later

Scripts largely overlap the predictive-control phases. `QualiaStructure` (what an object
is for, made of, and comes from) could feed the core's telos and the frame's expectations.

## 6. Generating the knowledge: two stages

| Stage | What the model produces | How often | Checked by |
|---|---|---|---|
| 1. Vocabulary (TBox) | The classes and properties of one perspective for one action concept | Once, then rarely | The rules below, HermiT against CRAM-AICOR, DUL and the core, and a person's review; then checked in, and the Python vocabulary generated from it as today |
| 2. Hypothesis (ABox) | The knowledge itself, as instances of that vocabulary | Once per situation, then stored (as today) | The existing acceptance, admission and repair |

Rules for a proposed vocabulary:

- Every new class or property specialises a CRAM-AICOR or DUL term, or a term of the core.
- An existing term is reused before a new one is made.
- Every new term has a definition (`rdfs:comment`), which becomes its docstring and the
  meaning the generation prompt shows.
- Roles, relations and parameter dimensions refer to the core's terms, never to another
  perspective's.
- Terms taken from FrameNet or ImageSchemaNet cite them with `rdfs:seeAlso`.
- Quantities carry a physical dimension and a unit.
- A property whose values are CRAM-AICOR concepts (an action category, a state type, a
  kind of object) names them by IRI and has no range, as the core's properties do.

Generation cost matters: one perspective took about three minutes in the real run. Each
perspective is generated separately, so the cost is paid once per perspective and
situation and can be paid ahead of time for the actions in use.

## 7. How the resolver uses it

- **Overview.** `knowledge_overview` answers with the core: the roles, the telos and
  effects, the parameter dimensions, and which perspectives the hypothesis holds. The
  drill-down tools answer per perspective: a role's expectations, the relations that
  constrain a dimension, the failures that concern it.
- **Roles to slots.** Next to the existing designator-to-structure mapping
  (`described_actions.json`), a per-designator mapping from the core's roles and parameter
  dimensions to the designator's slots (`SourceRole` ↔ `source_container`, *where
  relative to the goal* ↔ `pour_side_offset`). The model proposes it once from the
  designator's docstrings; it is accepted when it passes automatic checks and kept as
  data like the existing mapping (D9). The knowledge itself stays independent of any designator.
- **In the prompt.** Each slot's entry in `<slots>` names the role or dimension it maps to,
  so the model reads the relevant knowledge per slot instead of a whole overview.
- **Grounds.** A ground cites a role, an expectation or a relation, per slot, so `why`
  says which knowledge supports which choice.

## 8. Measuring it

On the kitchen cases, with and without each perspective: first-try success, the number of
trials and repair rounds, which slots are grounded in knowledge, and whether the
explanations name the right roles and relations. A perspective that does not change these
is not worth its generation cost.

## 9. Phases (each change test-first)

Pouring with one perspective goes all the way to the resolver and is measured before
another perspective is built (phase 4 decides whether to go on).

**Phase 0 — the base. Done 2026-10-15.**
- The action core for pouring (`action-core.ttl`, generated `action_core.py`).
- The vocabulary-proposal stage (§6, `vocabulary_proposal/`), run for the action-frame
  perspective of pouring with Sonnet 5.5 ($0.31, accepted on the first answer).
- The review of that proposal (`status.md`): postures and affordances named by CRAM-AICOR
  actions and states; CRAM-AICOR concepts named by IRI without a range, as in the core; the
  core's `hasQuantityKind` reused instead of copied.

**Phase 1 — register the action frame. Done 2026-10-15.**
1. The proposal writer writes the same Turtle for the same vocabulary, so a proposal and
   its review read as a plain git diff.
2. `action-frame.ttl` becomes an action perspective module (`ActionPerspectiveFile`), with
   its generated Python vocabulary and tests like the core's (structure, links to
   CRAM-AICOR and DUL, HermiT). The proposal's record stays under `proposals/`.

   Registering changed two things in the reviewed vocabulary. The frame's statements about
   an action frame hang off a `FrameStereotype` of it (`stereotypeOf` a core frame) rather
   than off the core's `ActionFrame`, so the core's Python holds nothing of any
   perspective and a hypothesis can hold any subset of them. Its polarity is
   `SlotConditionPolarity`, since Python names are unique across modules and the Flanagan
   core already has a `ConditionPolarity` meaning the same; phase 7 merges the two into the
   core.

**Phase 2 — knowledge in the new vocabularies (the hypothesis stage, §6).**
1. A hypothesis about pouring holding the core and the action frame: generated, accepted
   and admitted like the Flanagan ones, and stored under a key naming its perspectives.
2. `knowledge_overview` answers with the core (slots, objective, consequences, parameter
   dimensions) and which perspectives the hypothesis holds; a drill-down tool answers a
   slot's expectations.

**Phase 3 — roles to slots (§7).**
1. The mapping from the core's slots and parameter dimensions to `PouringAction`'s
   fields, proposed by the model from the designator's docstrings and the code ontology
   (`cram_classes`), accepted by automatic checks and kept as a file (D9).
2. Each slot's entry in the resolver's prompt names the slot or dimension it maps to.

**Phase 4 — measure (§8).** A live comparison on the pouring cases with and without the
action frame: first-try success, trials, repair rounds, grounded slots, explanations. Go on
only if the frame helps.

**Phase 5 — the image-schema perspective (§5.3). Built 2026-10-19**, through the same path:
proposed (Sonnet 5.5, $0.53, accepted on the second answer, HermiT consistent), reviewed and
registered as `image-schemas.ttl` (5 schema kinds, participations, requirements naming the
core's parameter dimensions and the quantities that decide them), its statements hanging off
a `SchemaStereotype`. Scripted knowledge tests pass. Measured with the failures in §13.6.

**Phase 6 — failures and corrections (§5.4). Built 2026-10-19**, the same way and seeded with
the failures measured on pouring (lands beside the opening, tilted too little to flow, the arm
cannot reach the pour pose; `pour_prediction.md` §5–§7): proposed for $0.50 (second answer,
HermiT consistent), reviewed and registered as `failures.ttl` under a `FailureStereotype`; a
correction is a change to a parameter dimension with a direction, a timing and what the new
choice must satisfy. Measured in §13.6 (arms `with_image_schemas`, `with_failures`,
`with_image_schemas_and_failures` in `experiments.frame_comparison`).

**Phase 7 — the Flanagan modules on the core's slots (§5.5)**, so every perspective talks
about the same participants; the Flanagan core's `ConditionPolarity` and the frame's
`SlotConditionPolarity` become one term of the action core.

**Phase 8 — more actions (D8):** navigating, picking up and placing in the core, with the
concepts CRAM-AICOR lacks proposed in our module (§10), and their perspectives.

## 10. Open questions

Answered 2026-10-15: the core's frame sits under `aicor:ManipulationFrame` (D6); one
perspective is proposed at a time (D7); navigating, picking up and placing come after
pouring (D8); a vocabulary is reviewed by a person, and a roles-to-slots mapping is accepted
by automatic checks (every slot maps to a role or a parameter dimension that exists; a
slot's type fits what the role expects, such as `Body` for a container role; units are
compatible) and kept as a file in the repository (D9).

1. **Navigating is not a manipulation.** CRAM-AICOR has `Grasping`, `Lifting`, `Reaching`
   and `Transporting` (an `ObjectDisplacement`), and `PlacementProblem` and
   `PlacementTarget` for placing, but no navigating concept, and `ManipulationFrame`
   describes manipulation problems. Proposed: a general `ActionFrame` ⊑ `dul:Description`,
   whose frames of manipulations are also `aicor:ManipulationFrame`s (D6 kept), with the
   navigating concept proposed in our own module until CRAM-AICOR has one.
2. **Picking up and placing as concepts.** CRAM-AICOR has no `PickingUp` or `Placing`
   action concept; they would be proposed in our module, built from the primitives it has
   (`Reaching`, `Grasping`, `Lifting`; `PlacementTarget`).

## 11. Phase 2 design (agreed 2026-10-15)

Decided: the core once with each perspective on top (§11.2); consulted beside the
Flanagan hypothesis (§11.6); one core and one hypothesis per perspective for each action
concept, with no situation features, until phase 4 shows a need for them.

### 11.1 What the pipeline already does for any vocabulary

| Stage | Generic | Tied to the Flanagan modules |
|---|---|---|
| Generation | The schema walks every type reachable from the root, with subclasses; decoding, patches and repair | The role and the content rule of the prompt describe predictive manipulation (phases, events, forces) |
| Acceptance | `RestrictionsHold`, `RangesHold` | `ShapesHold` and `ComponentsDeclared` apply only to Flanagan structures, so they find nothing to check elsewhere |
| Vocabulary lookup | | `Vocabulary` (IRI → generated class) knows only the Flanagan modules |
| Admission | HermiT, contradiction finding | `HermitCheck.from_environment` loads only the Flanagan modules |
| Store | Key: root type and fixed values | The key's module versions are those of every Flanagan module |
| Reading | `find_fragments`, `describe_fragment` | `HypothesisView.root` and `knowledge_overview` expect a phase structure |
| Resolver | | A designator maps to one Flanagan structure; a consultation serves one record, and the tools read one record |

### 11.2 Hypotheses: the core once, each perspective on top

```
core hypothesis          the(PouringFrame)(frames_action=aicor:Pouring)
  PouringFrame, its slots and parts, objective, consequences, parameter dimensions
        ▲ stereotype_of (a stored fragment, cited)
frame hypothesis         the(FrameStereotype)(stereotype_of=<core frame>)
  FrameStereotype, slot expectations, conditions, defaults, preconditions, rationales
```

- The **core hypothesis** is generated once per action and keyed as today.
- A **perspective hypothesis** is generated *given* a stored core hypothesis: the model
  is shown the core's fragments, refers to them by their ids and writes only the
  perspective's fragments. It is keyed by its root type and the core frame it cites, so
  it belongs to that core hypothesis.
- A consultation reads the core and any subset of its perspectives. Every perspective
  talks about the same slots, the core is paid for once, and phase 4 compares
  consultations that differ only in the perspective.

New: a fixed value that is a stored fragment, as built: the request fixes it to a
`Citation`, which the store keys by its IRI and the model is told by its name. The
backend reads the cited records (`HypothesisStore.given`) and the prompt shows their
fragments in a `<given>` section, written as the answer writes fragments; the answer
refers to them by name, and an answer naming a fragment like a given one, or stating a
fragment of the core, is returned for repair. The open-world check sees the given and the
new fragments together. The record lists the records it `builds_on`, its Turtle refers to
their fragments' IRIs without stating them again, and the store and the knowledge graph
read those references as the given fragments themselves.

Not proposed: one hypothesis holding the core and the frame. It needs nothing new, but
the core would be regenerated for every set of perspectives, so slots differ between
them, and the answer grows with each perspective.

### 11.3 The prompt's content per root

The role and the content rules come from the root's module (`generation/briefs.py`): the
Flanagan text for a phase structure; for the core or a perspective, its description,
kept beside the modules (`action_perspectives/descriptions/`), the text a perspective's
vocabulary is proposed from too.

### 11.4 Lookup, admission and the store key

- `Vocabulary` covers every registered module, both families.
- Each family says which modules a hypothesis rooted in it may use
  (`usable_modules`): every Flanagan module, which specialise one microworld; an action
  perspective and the core, so a core hypothesis is offered no class a perspective
  declares, not even a subclass of a core class such as `AdditionalConsequence`.
- The modules a hypothesis uses are those declaring the types its schema reaches. The
  configured check covers them (`ConsistencyCheck.covering`: HermiT adds the files it
  lacks), and the record keeps their versions, so a stored Flanagan record stays valid
  when an action perspective changes, and the reverse.
- The acceptance suite stays as it is.

### 11.5 Reading

- `knowledge_overview` by root: for a phase structure as today; for the core, the action
  concept, the frame, its slots with their parts, objectives with the state aimed at,
  consequences, parameter dimensions with their quantity kinds and slots, and the
  perspectives held.
- `describe_fragment` on a slot already lists the fragments linking to it, so a slot's
  expectations take one call; no new tool until phase 4 shows a need.
- The tools read every record of a consultation as one body of knowledge
  (`tools(*record_ids)`, with the records they build on); an action perspective
  hypothesis's fragments are named after its module (`action-core/source`), so they stay
  apart from each other and from the Flanagan fragments, whose names do not change.

### 11.6 Resolver

- A described action gains its core frame class (`core:PouringFrame`) beside its
  Flanagan structure (`described_actions.json`, `"frame"`).
- `consult` serves the Flanagan hypothesis as today, and, when the requester's
  `perspectives` name any (none by default, so runs are unchanged; phase 4 compares none
  with the action frame), the core and each perspective on it. A perspective's query
  names the stored core by its record (`StoredRoot`), resolved to the core frame's
  citation by the knowledge component, which knows where each perspective is rooted
  (`PERSPECTIVE_ROOTS`: the action frame's `FrameStereotype.stereotype_of`).
- The tools and the grounds read every record of the consultation; a second rule
  (`frame_knowledge`) says how to read the frame and its perspectives when they are
  offered. `KnowledgeConsulted` reports the records served for them.

### 11.7 Order (each test-first, with a scripted model until the last)

1. Lookup over every module, the modules a root uses (HermiT background, record
   versions), and the prompt's content per root; a core hypothesis generated, accepted,
   admitted, stored and read back. **Done** 2026-10-15; the Flanagan prompts are
   unchanged byte for byte.
2. Stored fragments as fixed values; a frame hypothesis on top of a core hypothesis.
   **Done** 2026-10-15.
3. The overview of a core hypothesis; tools over a consultation's records. **Done**
   2026-10-15 (knowledge-access contract 1.2).
4. The resolver consulting the core and the offered perspectives beside Flanagan.
   **Done** 2026-10-15, off by default.
5. Live with Sonnet 5.5: the core and the frame for pouring, the cost reported, the
   knowledge reviewed. **Done** 2026-10-15: both admitted on the first answer and found
   consistent by HermiT, $0.065 together (`status.md`).

## 12. Phase 3 design (agreed 2026-10-16)

Decided: the mapping is keyed by vocabulary terms (§12.1); it is written by hand for
`PouringAction`, checked automatically and reviewed as a diff, and a model proposes
mappings once navigating, picking up and placing need them (phase 8); the tolerances,
`arm` and `pour_side` are not mapped.

### 12.1 What a mapping says

For one designator, each field the action's frame speaks about, and what it speaks about
in the core's vocabulary, so the mapping holds for any core hypothesis of the action
(whose fragments, such as `action-core/p_pos`, are named anew each time it is
generated):

| Field holds | Maps to | `PouringAction` |
|---|---|---|
| An object filling a participant | A participant slot, by class | `source_container` → `SourceSlot`; `target_container` → `GoalSlot` |
| A choice the action leaves open | A parameter dimension, described by its quantity kind, the slot it is about (or a part of it) and, when it has one, the slot it is measured from | `tilt_angle` → orientation of the source; `pour_side_offset` → position of the source relative to the goal; `pour_height` → distance of the source from the goal; `pour_approach_offset` → position of the source relative to the goal |
| Anything else | Nothing | `position_threshold`, `orientation_threshold` (the controller's tolerances); `arm`, `pour_side` (which arm and side: the robot's, not the action's) |

The mapping is kept with the designator's other knowledge in `described_actions.json`.

### 12.2 Checked automatically, reviewed as a diff (D9)

Every mapped field exists on the designator (read from the code ontology,
`cram_classes`); every slot class and quantity kind exists in the core; a field holding an
object maps to a participant slot, a field holding a number to a dimension; a field maps
to one thing.

### 12.3 Matching a mapping to a core hypothesis

At a consultation, each mapped field is matched to the fragments of the consulted core
hypothesis: a slot class to its slots of that class; a dimension description to its
dimensions of that quantity kind about that slot or a part of it, and measured from that
slot when the description names one. A field matching nothing is named nothing. This is
the knowledge component's: the knowledge-access contract gains
`frame_parts_filled(record_id, frame_fields)`, the names of the fragments each field
fills, given the mapping the resolver holds (version 1.3; a component without frames
names none).

### 12.4 In the resolver's prompt

When perspectives are offered, each open slot's entry in `<slots>` names the core's slots
or dimensions it fills (`"frame": ["action-core/source"]`), and the `frame_knowledge`
rule says to read those first.

### 12.5 Order (each test-first)

1. The mapping in `described_actions.json`, its checks, and matching it to a core
   hypothesis. **Done** 2026-10-16 (`ontologies/frame_fields.py`).
2. Slot entries naming the fragments they fill. **Done** 2026-10-16.
3. The mapping for `PouringAction`, reviewed. Written 2026-10-16, for review: against the
   live core, `source_container` and `target_container` fill its source and goal,
   `tilt_angle` its orientation choice, `pour_side_offset` and `pour_approach_offset`
   its lip-position choice; `pour_height` fills nothing, that core having no distance
   choice. Reviewed 2026-10-16: kept as written, both offsets on the lip position; the
   core's description now names pouring's usual choices (tilt, lip position, height above
   the goal, amount), and a core generated with it fills all six fields ($0.069).

## 13. Phase 4 design (agreed 2026-10-16)

Decided: both measurements; three runs per arm of the simulated pour, and K1–K5 once per
arm; the comparison kept in `experiments/` so it can be rerun after each change and for
later perspectives. About $1.5–2.

The question: does the action frame, read beside today's Flanagan hypothesis, change what
the resolver does, and for the better? Each comparison runs the same cases with the
requester offering no perspective (today) and offering the action frame; nothing else
differs.

### 13.1 Where it is measured

| | Pouring in simulation (main) | The kitchen's knowledge cases |
|---|---|---|
| Cases | cram_viz's `pour_geometry_open` in `simple_pouring`: the cup and the bowl given, the arm, the tilt and the pour geometry open | K1–K5 of `experiments/llm_reasoner_kitchen`: one `PouringAction`, the arm and the tilt open, five tasks |
| What decides success | `SimulatedPouring`: each answer is poured in coraplex's simulation and where the pitcher's mouth ends is measured against the bowl's opening, up to three trials | No outcome; what was chosen and what it rests on |
| Why | The frame says the lip's position over the opening decides whether the substance lands in the goal; the baseline's first answer kept the defaults and missed by 0.188 m | Whether the frame changes choices and grounds where only the task differs |

### 13.2 What is recorded per run

First-try success (the mouth over the opening on the first trial), the trials and
repair rounds, the distance of the mouth from the opening's centre on each trial, the
values chosen, which grounds cite the frame (`action-core/…`, `action-frame/…`) or the
Flanagan hypothesis, which knowledge tools were called, and the cost.

### 13.3 Keeping the arms apart from generation

Both arms read stored knowledge: the Flanagan, core and frame hypotheses are generated
once into a baseline store before the runs, and every run consults a copy of it, so no
run pays for or varies by a generation.

### 13.4 Result of the simulated pour (2026-10-16)

No difference in outcome: no first pour landed in either arm, each run needed two pours.
The frame changed what the answers rest on (the core's lip position, tilt and height in
every run) and doubled the resolver's cost; the first answers kept the default geometry
in both arms, trusting coraplex's built-in sideways shift (`status.md`). The miss is a
question of where the lip ends for given values, which no stereotype answers.

The kitchen's K1–K5 (2026-10-16): the same choices in every case and arm; with the frame,
four of five answers rested on the core's tilt choice and the model read three to six
knowledge tools against one or two, for 30 % more. The frame changes grounding, not
choices, on these cases: phase 4's condition for going on is not met by them.

The containers open (P2, P3, P8, `objects_open`, twice per arm): both arms chose right in
every run; with the frame every kitchen answer rested on the frame's expectations of the
source and the goal, for about 20 % more. Verdict: on the pouring cases measured the frame
improves the grounding of answers, not their choices; the cases where a choice failed
failed on geometry, which no perspective describes.

Decided after phase 4 (2026-10-16): the action frame stays, opt-in, for the grounding it
gives answers; trap cases where common sense should decide (a closed carton or an
upside-down bowl as the only plausible goal) are measured later, before any further
perspective; the geometric miss is addressed by the resolver predicting where the pour
ends, without changing coraplex.

### 13.5 What is built for it

- `Reasoner` gains the requester's `perspectives` setting, so cram_viz and the kitchen
  can offer the frame (test-first).
- A comparison script running both arms and writing one record per run and a summary.

### 13.6 Image schemas and failures (2026-10-19)

Measured 2026-10-19 (Sonnet 5.5, $3.34 for 18 recorded runs; hypotheses for the new
perspectives generated into the store on first use, their cost not recorded):

| | Default | Image schemas | Failures | Both |
|---|---|---|---|---|
| `pour_geometry_open`, first pour lands | 0 of 3 | 0 of 3 | 0 of 2 | 0 of 3 |
| `pour_geometry_open`, any pour lands (two pours each) | 3 of 3 | 3 of 3 | 2 of 2 | 3 of 3 |
| `pour_geometry_open`, resolver cost per run | $0.102 | $0.126 | $0.142 | $0.125 |
| `objects_open`, any pour lands | 0 of 2 | 0 of 2 | 0 of 1 | 0 of 2 |
| `objects_open`, resolver cost per run | $0.242 | $0.261 | $0.422 | $0.285 |
| Grounds citing the core or a perspective | none | 10 (`pour_geometry_open`), 2 | 8, 3 | 9, 5 |

- Same verdict as the action frame: the perspectives change what the answers rest on, not
  what they choose. Every first pour kept the default geometry and missed by 0.188 m, and
  the corrected second pour landed within 5–19 mm in every arm; `objects_open` landed no
  pour in any arm.
- The new knowledge costs 20–70 % more per run, the failures most (it was read through up to
  15 tool calls).
- The failures arm failed in 2 of its 5 runs (one of each case): the resolver's own final
  turn, after many knowledge and world tool calls, was cut off at cram_viz's 8192 output
  tokens (`RESOLVER_MAX_TOKENS`); it was not the hypothesis generation. The run failed rather
  than being repaired. What the model wrote in that turn is not recorded.
- The tilt and reach failures (`pour_prediction.md` §5–§7) were not recovered by the
  resolver: `objects_open` tried tilts down to 1.0 and up to 2.7 rad and none landed.

Decided: both stay opt-in, like the frame.

### 13.7 A goal that cannot receive the pour (2026-10-20)

The first trap case: the scene's only bowl stands upside down
(`cram_viz.resolution.overturned_scene`), and the scene geometry in the prompt now states
which way each object's own up axis points (`"up": [0, 0, -1]` for the bowl), since an
upside-down bowl has the same bounding box as an upright one. `objects_open` leaves the goal
to the resolver, so the other bodies, among them a milk carton and a cereal box, are the
alternatives. Two runs of each arm (default, image schemas, failures, both), Sonnet 5.5,
$2.24 for the 7 runs that finished.

| | Poured into the overturned bowl | Refused or chose another goal | Cut off |
|---|---|---|---|
| Default | 2 of 2 | 0 | 0 |
| Image schemas | 1 of 2 | 0 | 1 (8192 output tokens) |
| Failures | 2 of 2 | 0 | 0 |
| Both | 2 of 2 | 0 | 0 |

No arm noticed. The prompt did state the bowl's up axis pointing down, and no answer mentions
it; the reasons are about the pour's lateral error. The image-schemas arm, whose vocabulary
says the lip must be above the opening, did not use it either. So on this case the
perspectives are not what is missing: the models do not read an up direction as an opening
facing down, and no check in the loop (the simulated pour measures the top of the bounding
box) can tell them. Two runs per arm is small, and a bare vector may be too indirect a fact;
a stated opening direction, or an instruction to check the goal's orientation, are untried.

## Sources

- M. Beetz, M. Kümpel. *From Frames to Pouring: The CRAM Cognitive Architecture for
  Everyday Robot Manipulation.* KI – Künstliche Intelligenz, 2026.
  doi:10.1007/s13218-026-00917-z
- A. Gangemi et al. *Framester: A Wide Coverage Linguistic Linked Data Hub.* EKAW 2016.
  doi:10.1007/978-3-319-49004-5_16
- S. De Giorgis, A. Gangemi, D. Gromann. *ImageSchemaNet: Formalizing Embodied Commonsense
  Knowledge Providing an Image-Schematic Layer to Framester.* Semantic Web, 2022.
  doi:10.3233/SW-223084
