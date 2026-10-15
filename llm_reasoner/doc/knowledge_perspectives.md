# Knowledge perspectives: design note

Status: **proposed** (2026-10-15), for review; nothing here is built. It refines
[`architecture.md`](architecture.md) §2.3 ("Intended role and design") and answers the
item *Representing knowledge for tools* in [`status.md`](status.md) §6.

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

**Phase 5 — the image-schema perspective (§5.3)**, through the same path: propose, review,
register, knowledge, compare.

**Phase 6 — failures and corrections (§5.4)**, the same way.

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

## Sources

- M. Beetz, M. Kümpel. *From Frames to Pouring: The CRAM Cognitive Architecture for
  Everyday Robot Manipulation.* KI – Künstliche Intelligenz, 2026.
  doi:10.1007/s13218-026-00917-z
- A. Gangemi et al. *Framester: A Wide Coverage Linguistic Linked Data Hub.* EKAW 2016.
  doi:10.1007/978-3-319-49004-5_16
- S. De Giorgis, A. Gangemi, D. Gromann. *ImageSchemaNet: Formalizing Embodied Commonsense
  Knowledge Providing an Image-Schematic Layer to Framester.* Semantic Web, 2022.
  doi:10.3233/SW-223084
