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
| The action and its frame | An `ActionFrame` ⊑ `aicor:ManipulationFrame`, linked to the action concept |
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
(`OpenContainerState`, `contains`), never in terms of a particular world. The resolver
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
  designator's docstrings; a person reviews it; it is kept as data like the existing
  mapping. The knowledge itself stays independent of any designator.
- **In the prompt.** Each slot's entry in `<slots>` names the role or dimension it maps to,
  so the model reads the relevant knowledge per slot instead of a whole overview.
- **Grounds.** A ground cites a role, an expectation or a relation, per slot, so `why`
  says which knowledge supports which choice.

## 8. Measuring it

On the kitchen cases, with and without each perspective: first-try success, the number of
trials and repair rounds, which slots are grounded in knowledge, and whether the
explanations name the right roles and relations. A perspective that does not change these
is not worth its generation cost.

## 9. Steps (each a separate change, test-first)

1. The action-core module for pouring, written against `cram-aicor-ontology.ttl`; the code
   generator and the existing checks run on it.
2. The vocabulary-proposal stage (§6) with its rules, run for the action-frame perspective
   of pouring; the proposal is reviewed and checked in.
3. Hypotheses holding the core and the action frame; `knowledge_overview` answering with
   the core.
4. The roles-to-slots mapping for `PouringAction`, and slot entries naming their roles.
5. A live comparison on the pouring cases, with and without the frame perspective.
6. The image-schema perspective, the same way; then failures and corrections.

## 10. Open questions

1. Where does the action core's `ActionFrame` sit: under `aicor:ManipulationFrame` (as
   proposed), or alongside `aicor:IntentionalStructure`, which `PouringIntention` already
   specialises with the participant roles and the telos?
2. Should the generation stage propose one perspective at a time, or the core and the
   frame together?
3. Is the roles-to-slots mapping reviewed by a person each time a designator is added, or
   accepted when it passes checks?
4. Which actions after pouring?

## Sources

- M. Beetz, M. Kümpel. *From Frames to Pouring: The CRAM Cognitive Architecture for
  Everyday Robot Manipulation.* KI – Künstliche Intelligenz, 2026.
  doi:10.1007/s13218-026-00917-z
- A. Gangemi et al. *Framester: A Wide Coverage Linguistic Linked Data Hub.* EKAW 2016.
  doi:10.1007/978-3-319-49004-5_16
- S. De Giorgis, A. Gangemi, D. Gromann. *ImageSchemaNet: Formalizing Embodied Commonsense
  Knowledge Providing an Image-Schematic Layer to Framester.* Semantic Web, 2022.
  doi:10.3233/SW-223084
