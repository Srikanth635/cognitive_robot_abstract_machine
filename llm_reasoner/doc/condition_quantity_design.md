# Conditions and quantities with one owner each (steps 13 and 14)

Design note of 2026-10-26. Steps 13 and 14 of the ontology consolidation
were deferred because the plan left four questions open: where polarity lives, how a list
of roles becomes the slots a condition or a quantity is about, how the physical dimensions
relate to the kinds of quantity, and whether the predictive manipulation (PM) terms
specialise the action core's or point to them. This note sets out each question with
options, a recommendation and what it costs.

**Decided (user, 2026-10-26):** P1, R1 with the condition table and
`ac:quantityInvolvesSlot`, D1 with `Magnitude`, `MaterialProperty` and `Ratio` as new
kinds, and S1. Both steps are built: P1 is met by a `SlotConditionType` above an
unchanged `SlotCondition`, an effector-object contact may also involve the grasp site, a
fourth new kind `Shape` covers `ExpectedShape`, and the unit `Ratio` became `Unitless`;
see [`status.md`](status.md).

## Why it matters

Today two families describe the same two notions:

| Notion | Action core (`ac:`) | Predictive manipulation core (`core:`) |
|---|---|---|
| A condition on participants | `SlotCondition`: on one slot, involving others, with a polarity | `PhaseConditionType`: on one or more roles, no polarity |
| A quantity about participants | `SlotQuantity`: about one slot, relative to at most one, of one kind, no value | `Parameter`: on one or more roles, of one physical dimension, with a value |

With one owner each, a perspective and a phase structure talk about the same conditions
and quantities, and the resolver can read a PM setpoint's value, such as a tilt range, for
the designator field that the frame's matching dimension (orientation about the source) maps to.
Competency questions can then cross the two families.

## What there is

**Conditions.** `ac:SlotCondition` requires `conditionOnSlot` (functional) and `hasPolarity`,
and may state `conditionInvolvesSlot`. Its subclasses are `StateCondition` (core), and
`AffordanceCondition`, `FeatureCondition` and `PostureCondition` (action frame); the action
frame's expectations and the failures' failed conditions use them. `core:PhaseConditionType ⊑
aicor:FluentType` requires `conditionOnRole some ManipulationRole` (not functional) and has
13 subclasses, from `ContactCondition` to `SubstanceFlowing`. A PM condition type is used through
`core:ConditionRequirement` (timing, polarity, world assumption, violation signal), and through
events and subgoals that establish it.

**Quantities.** `ac:SlotQuantity` requires `aboutSlot` (functional) and `hasQuantityKind`,
and may state `relativeToSlot` (functional); its subclasses are `ParameterDimension`,
`schemas:DecidingQuantity` and `failures:Indicator`. `core:Parameter` requires `hasValueForm`
and `parameterOnRole some ManipulationRole`; a quantitative one has a `MagnitudeRange` with a
unit and a `PhysicalDimension`, and twelve parameter classes are defined by their dimension
(`AngleParameter ≡ QuantitativeParameter ⊓ hasPhysicalDimension value Angle`). There are about
55 parameter classes. `MeasurementUnit` measures a `PhysicalDimension`.

**The data.** In the ten stored hypotheses (the cram_viz sample and the nine of
`~/.cram_viz/hypotheses`):

| | one role | two roles | more |
|---|---|---|---|
| Condition types (`conditionOnRole`) | 18 | 49 | 0 |
| Parameters (`parameterOnRole`) | 85 | 34 | 0 |

Conditions with two roles: `ObjectSupportContact` 8, `SubstanceContainedInSource`,
`GraspStability`, `SubstanceAccumulatedInTarget`, `LipOverReceptacle` and
`EffectorObjectContact` 6 each, `SubstanceFlowing` 4, `PhaseConditionType` itself 3,
`EffectorPositioned` 3, `ObjectClearOfSupport` 1. The recorded answers the tests replay name
roles for 2 conditions and 1 parameter, one role each. Only one shape reads the role lists
(`SafetyMarginOnOneParticipant`); the rest of the code reads them generically.

## Decision 1: where polarity lives

The two families model a condition in two shapes. In the action core a condition is a
statement with a polarity ("the source is not empty"). In the PM core a condition type is a
kind of world state ("substance contained in the source"), and the requirement on it says
when and whether it must hold.

- **P1, polarity on the statements that need it (recommended).** `ac:SlotCondition` keeps
  the slots and loses the polarity requirement. The subclasses that state a condition as
  holding or not (`StateCondition`, `AffordanceCondition`, `FeatureCondition`,
  `PostureCondition`) require it themselves, so the perspectives' fields, prompts and repairs
  stay as they are. `PhaseConditionType ⊑ ac:SlotCondition` needs no polarity, and
  `ConditionRequirement` keeps its own, as today.
- **P2, every PM condition type states a polarity.** The migration adds `Holds` to every PM
  condition, which then duplicates the requirement's polarity, and the two can disagree.
- **P3, perspectives state requirements like PM.** A perspective's condition becomes a type
  plus a requirement carrying the polarity. All three perspectives, their prompts and their
  recordings change.

## Decision 2: the role list

`conditionOnRole` and `parameterOnRole` list roles in no order; RDF keeps none. So the plan's
"the first role is the one the condition is on" cannot be decided from the records.

- **R1, each class says which role it is about (recommended).** A condition or parameter
  class restricts which slot it is on or about, and which slot it involves. The migration
  then splits a two-role record by the classes of its roles, deterministically. A record
  whose roles match no declaration stays stale: the three typed only `PhaseConditionType`,
  and any `ContactCondition` that is not one of its subclasses. Proposed for the conditions:

  | Condition | On | Involves |
  |---|---|---|
  | `EffectorObjectContact` | grasping effector | manipulandum |
  | `EffectorPositioned` | grasping effector | grasp site |
  | `GraspStability` | manipulandum | grasping effector |
  | `ObjectSupportContact`, `ObjectClearOfSupport` | manipulandum | support |
  | `SafeContactForce` | grasping effector | manipulandum |
  | `ObjectIntact` | manipulandum | — |
  | `LipOverReceptacle` | pouring lip | target receptacle |
  | `SourceUpright` | source container | — |
  | `SubstanceContainedInSource` | poured substance | source container |
  | `SubstanceAccumulatedInTarget` | poured substance | target receptacle |
  | `SubstanceFlowing` | poured substance | pouring lip |

  For the parameters, "relative to" fits a geometric quantity, but most two-role parameters
  quantify an interaction between two participants instead. `LipAlignmentSetpoint` (the lip
  relative to the receptacle) is geometric. The grip, load and slip forces, the grip-to-load
  ratio, the friction coefficient and the touchdown threshold all act *between* an effector or
  a manipulandum and another participant. R1 therefore also needs
  **`ac:quantityInvolvesSlot`**, a non-functional property beside `aboutSlot` and
  `relativeToSlot`, so an interaction quantity names its other party without posing as a
  reference frame.
- **R2, one non-functional "concerns" property.** A super-property
  `ac:conditionConcernsSlot` of `conditionOnSlot` and `conditionInvolvesSlot`, which PM states
  directly: a lossless rename, with no split. But the generated descriptors infer a
  super-property from its sub-properties and never let a model state it, so this needs a
  change to how inferred fields work. It also leaves PM conditions without the "on" slot that
  perspectives state.
- **R3, the first role is the "on" slot.** Rejected: there is no first.

## Decision 3: physical dimensions and kinds of quantity

The 7 kinds (`Amount`, `Rate`, `Position`, `Orientation`, `Distance`, `Duration`,
`Selection`) say *what about a slot* a quantity is. The 14 dimensions (`Mass`,
`MassFlowRate`, `Force`, `ForceRate`, `Torque`, `Length`, `Angle`, `AngularRate`, `Volume`,
`VolumetricFlowRate`, `DynamicViscosity`, `Frequency`, `Time`, `Dimensionless`) say *what it is
measured in*. These are two axes. An orientation is measured as an angle; a position or a
distance as a length; an amount as a volume or a mass; a rate as a mass, volume, angle or
force rate. Force, torque, viscosity, frequency and ratios fit none of the current kinds.

- **D1, two axes, both owned by the action core (recommended).** `PhysicalDimension`,
  `MeasurementUnit` and `measuresDimension` move into the action core (renames). Each kind
  names the dimensions it may be measured in (`ac:measuredIn`: Orientation in Angle; Amount in
  Mass and Volume; and so on). A quantity states both, and the unit check stays as precise as
  today. The 7 kinds need additions for what PM measures beyond them; proposed are
  `Magnitude` (a force or torque one participant exerts on another), `MaterialProperty`
  (viscosity, friction) and `Ratio` (grip to load). PM parameter classes fix their kind by
  restriction (`TiltAngleSetpoint ⊑ hasQuantityKind value Orientation`), so the value is
  entailed and the model is not asked for it.
- **D2, merge the dimensions into the kinds (your decision of 2026-10-25).** Keep one list.
  Where both name the same quantity keep one; add Force, Torque, DynamicViscosity, Frequency
  and Ratio as kinds. The mapping loses distinctions:

  | Dimension | Kind | Lost |
  |---|---|---|
  | Angle | Orientation | an angle that is no orientation, such as a lip alignment error |
  | Length | Distance or Position | which of the two |
  | Volume, Mass | Amount | volume against mass |
  | MassFlowRate, VolumetricFlowRate, AngularRate, ForceRate | Rate | which rate |
  | Time | Duration | — |

  A unit then measures a kind, so `Rate` accepts kg/s for a tilt rate: the unit check can no
  longer catch that error.
- **D3, keep both where they are, linked.** `core:PhysicalDimension` stays in PM, and a
  mapping says which kinds a dimension measures. Two owners remain.

## Decision 4: specialise or point

- **S1, PM specialises the core (the plan; recommended).** `PhaseConditionType ⊑
  ac:SlotCondition` and `Parameter ⊑ ac:SlotQuantity`; every PM fragment states its own slots.
  Since step 12 its roles are slots.
- **S2, PM points to the core.** A PM parameter *sets* a frame's `ParameterDimension`
  (`core:setsDimension`). The slots are then said once, by the frame, but a phase structure
  could only be generated after, and on top of, a core hypothesis. That changes how knowledge
  is requested.

## What it costs

| | Recommended (P1, R1, D1, S1) |
|---|---|
| Ontology | action core: polarity requirement moved, `quantityInvolvesSlot`, the dimension partition and units, `measuredIn`, three kinds; PM core: condition and parameter classes specialise the core's; the 13 condition classes and the 15 or so parameter classes measured between two participants declare their slots, and every parameter class fixes its kind; `conditionOnRole`, `parameterOnRole` and `hasPhysicalDimension` go |
| Migration | split by the roles' classes: 49 conditions and 34 parameters in the stored records; the 3 bare `PhaseConditionType` with two roles stay stale; renames for the dimensions and units |
| Prompts | PM answers state `condition_on_slot` and `condition_involves_slot`, and `about_slot`, `relative_to_slot` and `quantity_involves_slot`; kinds are entailed. A live check per root is needed, and is blocked until the API credits are topped up |
| Tests | `test_slot_condition`, `test_slot_quantity`, `test_acceptance`'s requirement and parameter helpers, the dimension and unit tests of `test_ontology_translation`, the recorded answers naming roles, the sample |
| Order | step 13 (P1, R1 for conditions) and step 14 (D1, R1 for quantities), one commit each, test-first, a live check after each |

## To decide

1. Polarity: P1, P2 or P3.
2. Role lists: R1 (with the condition table above, and `quantityInvolvesSlot`) or R2.
3. Dimensions: D1 (and the names of the new kinds), D2 or D3.
4. Specialise or point: S1 or S2.
