"""Flanagan/Johansson action-phase microworld: the action-neutral core.

Manipulation tasks decompose into discrete action phases, each driven by a
SUBGOAL, parameterised in a feedforward manner by INTERNAL MODELS of object
properties, and delimited by DISCRETE MECHANICAL EVENTS that serve as
sensorimotor control points where predicted and actual sensory signals are
compared.

SCOPE AND EPISTEMIC STATUS
--------------------------
Everything here is at the DESCRIPTION level: stereotypical, type-level,
non-indexical knowledge about what a phase structure for a kind of manipulation
looks like in general. It does not represent occurrences. Instances of these
classes are PROTOTYPES, never episodes, and must be kept in a knowledge scope
disjoint from the episodic world model.

ACTION-NEUTRALITY
-----------------
This module names no particular phase, no particular event and no particular
grasp. Anything answering "what are the phases of THIS kind of action" belongs
in an extension module, so a second action can be added without touching this
file.

ALIGNMENT
---------
``aligns_to`` records the DUL/SOMA term a class corresponds to. Alignment is
annotation only: it is exported faithfully, but no foreign axioms are imported
and none are reasoned with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Optional, assert_never

from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol

from llm_reasoner.knowledge.ontological.decorators import (
    ConceptReference,
    KnowledgeEntity,
    ontological,
    prop,
    value_partition,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel, Rigidity

FLAN = "http://example.org/ontology/flanagan-core#"
DUL = "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"
SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"


# %% value partitions
# Orthogonal dimensions, so enums rather than sibling classes. Modelling them
# as siblings is what forced the original ontology into making a continuously
# modulated phase a sibling of a coupled one -- when a pouring tilt phase is
# both.


@value_partition(FLAN + "CouplingRegime")
class CouplingRegime(StrEnum):
    """Whether effector and manipulandum form a single kinematic unit."""

    UNCOUPLED = "uncoupled"
    """Effector moves without being mechanically bound to the object."""
    TRANSITIONAL = "transitional"
    """Contact or force conditions are changing but the object does not
    translate."""
    COUPLED = "coupled"
    """Effector and object move together as one kinematic unit."""


@value_partition(FLAN + "RegulationMode")
class RegulationMode(StrEnum):
    """How control commands are regulated within a phase."""

    FEEDFORWARD = "feedforward"
    """Commands are selected from predictions before sensory correction."""
    CONTINUOUS_FEEDBACK = "continuous_feedback"
    """Commands are continuously corrected from monitored sensory signals."""
    HYBRID = "hybrid"
    """Feedforward predictions and continuous sensory correction cooperate."""


@value_partition(FLAN + "ComparisonOutcome")
class ComparisonOutcome(StrEnum):
    """Whether an observation agrees with its sensory prediction."""

    MATCH = "match"
    """The observation supports the prediction within its declared tolerances."""
    MISMATCH = "mismatch"
    """The observation differs from the prediction in a classified way."""


@value_partition(FLAN + "PredictionErrorKind")
class PredictionErrorKind(StrEnum):
    """The dimension along which an observation violates a prediction."""

    ABSENT = "absent"
    """The predicted event or feature was not observed."""
    UNEXPECTED = "unexpected"
    """An event or feature outside the prediction was observed."""
    EARLY = "early"
    """The predicted event or feature occurred before its timing window."""
    LATE = "late"
    """The predicted event or feature occurred after its timing window."""
    MAGNITUDE_MISMATCH = "magnitude_mismatch"
    """The observed magnitude fell outside the prediction and tolerance."""
    DIRECTION_MISMATCH = "direction_mismatch"
    """The observed change proceeded in the opposite direction."""
    SPATIAL_MISMATCH = "spatial_mismatch"
    """The observation involved a different location or participant."""


@value_partition(FLAN + "CorrectionDirection")
class CorrectionDirection(StrEnum):
    """How a corrective response changes its target setpoint."""

    INCREASE = "increase"
    """Raise the target setpoint."""
    DECREASE = "decrease"
    """Lower the target setpoint."""
    HOLD = "hold"
    """Maintain the target at its present value."""
    REVERSE = "reverse"
    """Command change opposite to the current direction."""
    STOP = "stop"
    """Bring the commanded change to rest."""


@value_partition(FLAN + "RecoveryMode")
class RecoveryMode(StrEnum):
    """How execution proceeds after a corrective response."""

    CONTINUE_PHASE = "continue_phase"
    """Apply the correction without restarting the phase."""
    REPEAT_PHASE = "repeat_phase"
    """Restart the current phase with corrected control parameters."""
    REPLAN = "replan"
    """Select a new phase sequence from the updated state."""
    ABORT = "abort"
    """Terminate the manipulation safely."""


@value_partition(FLAN + "EstimateUpdateDirection")
class EstimateUpdateDirection(StrEnum):
    """How evidence changes a represented estimate."""

    INCREASE = "increase"
    """Revise the estimate upward."""
    DECREASE = "decrease"
    """Revise the estimate downward."""
    REVISE = "revise"
    """Change a non-ordered estimate or replace its current value."""


@value_partition(FLAN + "ContactTransition")
class ContactTransition(StrEnum):
    """The contact-topology change carried by a mechanical event."""

    ESTABLISHES = "establishes"
    """The event creates contact between its participants."""
    BREAKS = "breaks"
    """The event removes contact between its participants."""


@value_partition(FLAN + "PhysicalDimension")
class PhysicalDimension(StrEnum):
    """A physical dimension used to validate quantitative knowledge."""

    MASS = "mass"
    """Quantity of matter."""
    MASS_FLOW_RATE = "mass_flow_rate"
    """Mass transferred per unit time."""
    FORCE = "force"
    """Mechanical force, including weight force."""
    FORCE_RATE = "force_rate"
    """Rate of force change."""
    TORQUE = "torque"
    """Moment of force."""
    LENGTH = "length"
    """Spatial distance or position component."""
    ANGLE = "angle"
    """Planar or rotational angle."""
    ANGULAR_RATE = "angular_rate"
    """Rate of angular change."""
    VOLUME = "volume"
    """Three-dimensional capacity."""
    VOLUMETRIC_FLOW_RATE = "volumetric_flow_rate"
    """Volume transferred per unit time."""
    DYNAMIC_VISCOSITY = "dynamic_viscosity"
    """Resistance of a fluid to shear flow."""
    FREQUENCY = "frequency"
    """Repetitions of a periodic signal per unit time."""
    TIME = "time"
    """Temporal duration or offset."""
    DIMENSIONLESS = "dimensionless"
    """A ratio or fraction without a physical unit."""


@value_partition(FLAN + "MeasurementUnit")
class MeasurementUnit(StrEnum):
    """A supported measurement unit with one fixed physical dimension."""

    KILOGRAM = "kg"
    """Kilogram, a unit of mass."""
    KILOGRAM_PER_SECOND = "kg/s"
    """Kilogram per second, a unit of mass flow rate."""
    NEWTON = "N"
    """Newton, a unit of force."""
    NEWTON_PER_SECOND = "N/s"
    """Newton per second, a unit of force rate."""
    NEWTON_METRE = "N*m"
    """Newton metre, a unit of torque."""
    METRE = "m"
    """Metre, a unit of length."""
    DEGREE = "deg"
    """Degree, a unit of angle."""
    DEGREE_PER_SECOND = "deg/s"
    """Degree per second, a unit of angular rate."""
    MILLILITRE = "mL"
    """Millilitre, a unit of volume."""
    MILLILITRE_PER_SECOND = "mL/s"
    """Millilitre per second, a unit of volumetric flow rate."""
    CENTIPOISE = "cP"
    """Centipoise, a unit of dynamic viscosity."""
    HERTZ = "Hz"
    """Hertz, a unit of frequency."""
    SECOND = "s"
    """Second, a unit of time."""
    MILLISECOND = "ms"
    """Millisecond, a unit of time."""
    DIMENSIONLESS = "dimensionless"
    """Unit marker for dimensionless ratios."""
    FRACTION = "fraction"
    """Unit marker for bounded dimensionless fractions."""

    @property
    def physical_dimension(self) -> PhysicalDimension:
        """Return the physical dimension represented by this unit."""
        match self:
            case MeasurementUnit.KILOGRAM:
                return PhysicalDimension.MASS
            case MeasurementUnit.KILOGRAM_PER_SECOND:
                return PhysicalDimension.MASS_FLOW_RATE
            case MeasurementUnit.NEWTON:
                return PhysicalDimension.FORCE
            case MeasurementUnit.NEWTON_PER_SECOND:
                return PhysicalDimension.FORCE_RATE
            case MeasurementUnit.NEWTON_METRE:
                return PhysicalDimension.TORQUE
            case MeasurementUnit.METRE:
                return PhysicalDimension.LENGTH
            case MeasurementUnit.DEGREE:
                return PhysicalDimension.ANGLE
            case MeasurementUnit.DEGREE_PER_SECOND:
                return PhysicalDimension.ANGULAR_RATE
            case MeasurementUnit.MILLILITRE:
                return PhysicalDimension.VOLUME
            case MeasurementUnit.MILLILITRE_PER_SECOND:
                return PhysicalDimension.VOLUMETRIC_FLOW_RATE
            case MeasurementUnit.CENTIPOISE:
                return PhysicalDimension.DYNAMIC_VISCOSITY
            case MeasurementUnit.HERTZ:
                return PhysicalDimension.FREQUENCY
            case MeasurementUnit.SECOND | MeasurementUnit.MILLISECOND:
                return PhysicalDimension.TIME
            case MeasurementUnit.DIMENSIONLESS | MeasurementUnit.FRACTION:
                return PhysicalDimension.DIMENSIONLESS
        assert_never(self)


@value_partition(FLAN + "ParameterValueForm")
class ParameterValueForm(StrEnum):
    """Whether a parameter value is quantitative or categorical."""

    QUANTITATIVE = "quantitative"
    """The parameter requires a :class:`MagnitudeRange`."""
    CATEGORICAL = "categorical"
    """The parameter requires a :class:`CategoricalValue`."""


@value_partition(FLAN + "TrajectoryTrend")
class TrajectoryTrend(StrEnum):
    """The direction of change expected across a signal trajectory."""

    INCREASING = "increasing"
    """The signal magnitude rises over the prediction interval."""
    DECREASING = "decreasing"
    """The signal magnitude falls over the prediction interval."""
    STABLE = "stable"
    """The signal remains within a bounded steady range."""


@value_partition(FLAN + "ConditionMode")
class ConditionMode(StrEnum):
    """When, relative to a phase, a condition is demanded."""

    AT_ENTRY = "at_entry"
    """Must hold for the phase to begin. The classical precondition."""
    THROUGHOUT = "throughout"
    """Must hold continuously for the duration. An invariant, not a
    precondition: grasp stability during transport is checked at every instant,
    and its violation is a failure event rather than a failed entry test."""
    AT_EXIT = "at_exit"
    """Is brought about by successful completion. The classical
    postcondition."""


@value_partition(FLAN + "ConditionPolarity")
class ConditionPolarity(StrEnum):
    """Whether a condition is demanded to hold or to be absent.

    Carried on the requirement rather than the condition type, so that
    "receptacle full" is a single condition that one phase may require and
    another forbid, instead of two unrelated conditions with no way to relate
    them.
    """

    MUST_HOLD = "must_hold"
    MUST_NOT_HOLD = "must_not_hold"


# %% participant roles: slots versus bound roles
# The distinction that took a wrong turn earlier and is worth stating plainly.
#
# krrood's Role[T] is TOKEN-LEVEL by construction: `role_taker` is required, so
# a role always has something bearing it. That is the right design and it is
# used here as intended -- but only on the far side of resolution.
#
# A hypothesis role is an UNBOUND SLOT. Nothing fills it yet; binding is what
# the resolver does later against the world model. Forcing a taker at this
# level meant inventing a world entity to satisfy the constructor, which put an
# indexical body inside knowledge that is meant to be generic.
#
# So there are two things, and only one taxonomy:
#   RoleSlot           -- description-level, unbound, appears in hypotheses,
#                         and names the bound-role class to instantiate.
#   ManipulationRole   -- a real krrood Role[T], constructed by the resolver
#                         when the slot is bound to a world entity.
#
# The slot carries the descriptive content; the Role classes carry the
# taxonomy. No duplication, and krrood's role machinery -- taker chains,
# has_role, roles_for, IsSameSemanticEntity -- works exactly as designed at the
# point where a taker actually exists.


@ontological(
    iri=FLAN + "ManipulationRole",
    aligns_to=DUL + "Role",
    rigidity=Rigidity.ANTI_RIGID,
    role_taker="Symbol",
    level=ConceptLevel.OCCURRENCE,
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class ManipulationRole(Role[Symbol]):
    """A participant role borne by an actual world entity.

    Occurrence-level: instances exist only after resolution has bound a slot to
    something. Anti-rigid, so pure composition -- a body that stops being the
    manipulandum is still that body.
    """


@ontological(
    iri=FLAN + "Manipulandum",
    aligns_to=SOMA + "MovedObject",
    rigidity=Rigidity.ANTI_RIGID,
    role_taker="Symbol",
    level=ConceptLevel.OCCURRENCE,
)
@dataclass(eq=False)
class Manipulandum(ManipulationRole):
    """The object whose state the manipulation is about."""


@ontological(
    iri=FLAN + "Actor",
    aligns_to=SOMA + "AgentRole",
    rigidity=Rigidity.ANTI_RIGID,
    role_taker="Symbol",
    level=ConceptLevel.OCCURRENCE,
    disjoint_with=["Manipulandum"],
)
@dataclass(eq=False)
class Actor(ManipulationRole):
    """The agent performing the manipulation.

    Disjoint from Manipulandum: an agent is never the object it manipulates.
    """


@ontological(iri=FLAN + "RoleSlot", display_name="Role")
@dataclass(eq=False)
class RoleSlot(KnowledgeEntity):
    """An unbound participant slot in a phase structure.

    This is what a hypothesis contains. It says which kind of role will be
    borne, and by how many entities, without naming any. Resolution consists of
    binding these to world entities, which is where the corresponding
    ManipulationRole instances come into existence.
    """

    binds_as: Optional[ConceptReference] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "bindsAs",
        concept_root=ManipulationRole,
        comment="The ManipulationRole class an instance of this slot becomes "
        "once bound. A reference rather than a type, so that slots "
        "persist and serialise like every other fragment.",
    )
    min_bindings: int = prop(
        functional=True,
        default=1,
        comment="How many world entities must fill this slot. One for a "
        "manipulandum; more where a role is collective.",
    )


# %% boundary events


@ontological(
    iri=FLAN + "PhaseBoundaryEventType",
    aligns_to=DUL + "EventType",
    abstract=True,
)
@dataclass(eq=False)
class PhaseBoundaryEventType(KnowledgeEntity):
    """A discrete transition that can terminate a phase."""

    is_failure_event: bool = prop(
        functional=True,
        default=False,
        iri=FLAN + "isFailureEvent",
        comment="Whether the event marks failure rather than intended completion.",
    )
    """Whether the event marks failure rather than intended completion."""


@ontological(iri=FLAN + "ExternalTerminationEventType")
@dataclass(eq=False)
class ExternalTerminationEventType(PhaseBoundaryEventType):
    """A phase boundary imposed by a typed source outside the manipulation."""

    triggered_by: Optional[ConceptReference] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "triggeredBy",
    )
    """The external request, policy, or safety condition ending the phase."""


@ontological(iri=FLAN + "MechanicalEventType")
@dataclass(eq=False)
class MechanicalEventType(PhaseBoundaryEventType):
    """A discrete physical transition the controller can predict and detect."""

    event_on_role: list[RoleSlot] = prop(
        min_count=1,
        default_factory=list,
        iri=FLAN + "eventOnRole",
        comment="The participants between which the event occurs.",
    )
    """The participants whose physical relation changes at the event."""


@ontological(iri=FLAN + "ContactEventType")
@dataclass(eq=False)
class ContactEventType(MechanicalEventType):
    """An event carrying an explicit contact-topology transition."""

    contact_transition: Optional[ContactTransition] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasContactTransition",
    )
    """Whether contact is established or broken."""


@ontological(iri=FLAN + "ThresholdCrossingEventType")
@dataclass(eq=False)
class ThresholdCrossingEventType(MechanicalEventType):
    """An event carrying an explicit mechanical-threshold crossing."""

    crosses_threshold: Optional[MechanicalThreshold] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "crossesThreshold",
    )
    """The threshold crossed at this event."""


@ontological(iri=FLAN + "ForceDynamicEventType")
@dataclass(eq=False)
class ForceDynamicEventType(ThresholdCrossingEventType):
    """A threshold-crossing event driven by contact or inertial force dynamics."""


@ontological(iri=FLAN + "ControlEventType")
@dataclass(eq=False)
class ControlEventType(MechanicalEventType):
    """An event carrying one or more attained controller or task setpoints."""

    attains_setpoint: list[Setpoint] = prop(
        min_count=1,
        default_factory=list,
        iri=FLAN + "attainsSetpoint",
    )
    """The setpoints whose attainment constitutes this event."""


@ontological(iri=FLAN + "SetpointAttained", disjoint_siblings=True)
@dataclass(eq=False)
class SetpointAttained(ControlEventType):
    """A commanded pose, orientation, force, or task quantity has been reached."""


# %% conditions and subgoals


@ontological(
    iri=FLAN + "PhaseConditionType",
    aligns_to=SOMA + "StateType",
    disjoint_siblings=True,
)
@dataclass(eq=False)
class PhaseConditionType(KnowledgeEntity):
    """A type of world state relevant to a phase.

    There are deliberately no Precondition/Postcondition subclasses. Being a
    precondition is relational and anti-rigid: the very same state type is the
    postcondition of one phase and the precondition of the next. Subclassing
    would force a spurious disjointness and duplicate every state type. The
    distinction is carried by ConditionRequirement instead.
    """

    condition_on_role: list[RoleSlot] = prop(
        min_count=1,
        default_factory=list,
        iri=FLAN + "conditionOnRole",
        comment="The roles the condition predicates over. Without these it is "
        "unbindable by the resolver.",
    )
    violation_signalled_by: Optional[MechanicalEventType] = prop(
        functional=True,
        iri=FLAN + "violationSignalledBy",
        comment="The failure event registering this condition ceasing to hold. "
        "Gives invariants operational content: an invariant nothing can "
        "detect the violation of is not a guard, merely a hope.",
    )


@ontological(iri=FLAN + "Subgoal")
@dataclass(eq=False)
class Subgoal(PhaseConditionType):
    """The mechanical or state objective an individual phase exists to achieve.

    In this framework a subgoal is exactly that postcondition whose attainment
    is signalled by the phase's terminating mechanical event. That is what makes
    the event a sensorimotor control point rather than an incident.
    """

    attainment_signalled_by: Optional[PhaseBoundaryEventType] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "attainmentSignalledBy",
        comment="The event whose registration constitutes evidence the subgoal "
        "is reached. Functional: a subgoal with two independent "
        "signatures is two subgoals. Must not be a failure event.",
    )


@ontological(
    iri=FLAN + "ConditionRequirement",
    aligns_to=DUL + "Relation",
    display_name="Condition",
)
@dataclass(eq=False)
class ConditionRequirement(KnowledgeEntity):
    """A phase's demand on a condition, qualified by mode, polarity and status.

    Reified so that mode, polarity and epistemic status are expressible without
    a combinatorial grid of properties. The alternative -- requires_condition,
    maintains_condition, requires_absence_of, assumes_condition and their
    cross-products -- is the same mistake as the Precondition/Postcondition
    subclasses this model already rejects.
    """

    on_condition: Optional[PhaseConditionType] = prop(
        functional=True, min_count=1, iri=FLAN + "onCondition"
    )
    mode: Optional[ConditionMode] = prop(
        functional=True, min_count=1, iri=FLAN + "hasConditionMode"
    )
    polarity: ConditionPolarity = prop(
        functional=True,
        default=ConditionPolarity.MUST_HOLD,
        iri=FLAN + "hasPolarity",
    )
    is_world_assumption: bool = prop(
        functional=True,
        default=False,
        marks_assumption=True,
        iri=FLAN + "isWorldAssumption",
        comment="True when no phase establishes this condition and it is "
        "presupposed of the world. The most useful flag for the "
        "resolver: assumptions are exactly what it must check against "
        "the world model, and what it must report when it cannot.",
    )


# %% parameter values


@ontological(
    iri=FLAN + "MagnitudeRange",
    aligns_to=DUL + "Region",
    display_name="Range",
)
@dataclass(eq=False)
class MagnitudeRange(KnowledgeEntity):
    """An interval-valued quantitative region with an explicit dimension and unit."""

    lower_bound: Optional[float] = prop(functional=True, min_count=1)
    """The inclusive lower bound."""
    upper_bound: Optional[float] = prop(functional=True, min_count=1)
    """The inclusive upper bound."""
    physical_dimension: Optional[PhysicalDimension] = prop(
        functional=True,
        iri=FLAN + "hasPhysicalDimension",
    )
    """The dimension shared by both bounds, inferred from :attr:`unit` when omitted."""
    unit: Optional[MeasurementUnit] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasMeasurementUnit",
    )
    """The measurement unit shared by both bounds."""

    def __setattr__(self, name: str, value: object) -> None:
        """Set a field and infer an omitted dimension from a recognized unit."""
        if (
            name == "unit"
            and isinstance(value, MeasurementUnit)
            and self.physical_dimension is None
        ):
            self.physical_dimension = value.physical_dimension
        super().__setattr__(name, value)


@ontological(iri=FLAN + "TemporalWindow", aligns_to=DUL + "Region")
@dataclass(eq=False)
class TemporalWindow(KnowledgeEntity):
    """An expected or observed time interval relative to a control event."""

    earliest_offset: Optional[float] = prop(functional=True, min_count=1)
    """The inclusive beginning of the interval."""
    latest_offset: Optional[float] = prop(functional=True, min_count=1)
    """The inclusive end of the interval."""
    unit: Optional[MeasurementUnit] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasMeasurementUnit",
    )
    """The time unit shared by both offsets."""


@ontological(iri=FLAN + "CategoricalValue")
@dataclass(eq=False)
class CategoricalValue(KnowledgeEntity):
    """A non-numeric value represented by a typed concept reference."""

    category: Optional[ConceptReference] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasCategory",
    )
    """The concept identifying the represented category."""


# %% parameter categories


@ontological(
    iri=FLAN + "Parameter",
    aligns_to=DUL + "Parameter",
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class Parameter(KnowledgeEntity):
    """A quantitative or categorical slot in a phase structure."""

    required_dimension: ClassVar[Optional[PhysicalDimension]] = None
    """The physical dimension required by this parameter type, when quantitative."""
    permitted_dimensions: ClassVar[Optional[frozenset[PhysicalDimension]]] = None
    """Alternative physical dimensions accepted by this parameter type."""
    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.QUANTITATIVE
    """The value form required by this parameter type."""

    parameter_range: Optional[MagnitudeRange] = prop(
        functional=True,
        iri=FLAN + "parameterRange",
    )
    """The quantitative value, mutually exclusive with :attr:`categorical_value`."""
    categorical_value: Optional[CategoricalValue] = prop(
        functional=True,
        iri=FLAN + "categoricalValue",
    )
    """The categorical value, mutually exclusive with :attr:`parameter_range`."""
    parameter_on_role: list[RoleSlot] = prop(
        min_count=1,
        default_factory=list,
        iri=FLAN + "parameterOnRole",
    )
    """The participant roles to which the parameter applies."""
    parameter_in_phase: list[ActionPhase] = prop(
        default_factory=list,
        iri=FLAN + "parameterInPhase",
    )
    """The phases in which the parameter is active."""


@ontological(
    iri=FLAN + "Estimate",
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class Estimate(Parameter):
    """A belief used to predict or scale manipulation control."""

    scales: list[MotorSetpoint] = prop(
        default_factory=list,
        iri=FLAN + "scales",
        comment="The motor setpoints scaled feedforward by this estimate.",
    )
    """The motor setpoints scaled feedforward by this estimate."""


@ontological(iri=FLAN + "ObjectPropertyEstimate", abstract=True)
@dataclass(eq=False)
class ObjectPropertyEstimate(Estimate):
    """A belief about a comparatively stable property of an object."""


@ontological(iri=FLAN + "ContactPropertyEstimate", abstract=True)
@dataclass(eq=False)
class ContactPropertyEstimate(Estimate):
    """A belief about a property of an effector-object contact interface."""


@ontological(iri=FLAN + "ObjectStateEstimate", abstract=True)
@dataclass(eq=False)
class ObjectStateEstimate(Estimate):
    """A belief about a changeable state of an object."""


@ontological(iri=FLAN + "DynamicsEstimate", abstract=True)
@dataclass(eq=False)
class DynamicsEstimate(Estimate):
    """A belief about how an object state evolves under action."""


@ontological(
    iri=FLAN + "Setpoint",
    aligns_to=SOMA + "Setpoint",
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class Setpoint(Parameter):
    """A desired value whose attainment can constitute a control event."""


@ontological(iri=FLAN + "MotorSetpoint", abstract=True)
@dataclass(eq=False)
class MotorSetpoint(Setpoint):
    """A target directly commanded by the motor controller."""

    adapts_to: list[Estimate] = prop(
        default_factory=list,
        iri=FLAN + "adaptsTo",
        comment="The estimates that continuously rescale this command.",
    )
    """The estimates that continuously rescale this command."""


@ontological(iri=FLAN + "TaskSetpoint", abstract=True)
@dataclass(eq=False)
class TaskSetpoint(Setpoint):
    """A desired task outcome rather than a direct motor command."""


@ontological(iri=FLAN + "ForceRateSetpoint")
@dataclass(eq=False)
class ForceRateSetpoint(MotorSetpoint):
    """A commanded rate of force change."""

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.FORCE_RATE
    """Force rate is the required physical dimension."""

    rate_of: Optional[MotorSetpoint] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "rateOf",
    )
    """The force setpoint whose rate is commanded."""


@ontological(
    iri=FLAN + "MechanicalThreshold",
    aligns_to=SOMA + "Threshold",
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class MechanicalThreshold(Parameter):
    """A boundary value whose crossing constitutes a mechanical event."""


@ontological(iri=FLAN + "SafetyMargin")
@dataclass(eq=False)
class SafetyMargin(Parameter):
    """The excess of a setpoint over a threshold it must remain clear of."""

    margin_on: Optional[MotorSetpoint] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "marginOn",
    )
    """The motor setpoint carrying the safety margin."""
    margin_over: Optional[MechanicalThreshold] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "marginOver",
    )
    """The threshold below the setpoint."""


# %% internal models and sensory signals


@ontological(
    iri=FLAN + "SensorySignalType", aligns_to=DUL + "EventType", disjoint_siblings=True
)
@dataclass(eq=False)
class SensorySignalType(KnowledgeEntity):
    """A type of afferent signal that can be predicted and compared at a control
    point."""


@ontological(iri=FLAN + "TactileSignal")
@dataclass(eq=False)
class TactileSignal(SensorySignalType):
    """Fingertip afferent signals; the primary channel for contact events."""


@ontological(iri=FLAN + "ProprioceptiveSignal")
@dataclass(eq=False)
class ProprioceptiveSignal(SensorySignalType):
    """Signals of limb configuration and effort; complements tactile signals
    during coupled phases."""


@ontological(iri=FLAN + "VisualSignal")
@dataclass(eq=False)
class VisualSignal(SensorySignalType):
    """Exteroceptive visual signals, used predictively to anticipate contact
    events before they occur."""


@ontological(iri=FLAN + "AuditorySignal")
@dataclass(eq=False)
class AuditorySignal(SensorySignalType):
    """Acoustic evidence about an event or continuously changing state."""


@ontological(
    iri=FLAN + "InternalModel",
    aligns_to=DUL + "Description",
    abstract=True,
)
@dataclass(eq=False)
class InternalModel(KnowledgeEntity):
    """The controller's own representation of body and environment, used to
    predict and to invert."""


@ontological(iri=FLAN + "ForwardModel")
@dataclass(eq=False)
class ForwardModel(InternalModel):
    """Predicts sensory consequences from outgoing motor commands; supplies the
    predictions compared at control points."""


@ontological(iri=FLAN + "InverseModel")
@dataclass(eq=False)
class InverseModel(InternalModel):
    """Determines the motor command required to reach a desired state."""


@ontological(
    iri=FLAN + "EstimateRepresentation",
    aligns_to=DUL + "Description",
    abstract=True,
)
@dataclass(eq=False)
class EstimateRepresentation(InternalModel):
    """Represented content that explicitly owns one or more estimates."""

    holds_estimate: list[Estimate] = prop(
        default_factory=list, iri=FLAN + "holdsEstimate"
    )
    """The estimates stored as content of the representation."""


@ontological(iri=FLAN + "ObjectRepresentation")
@dataclass(eq=False)
class ObjectRepresentation(EstimateRepresentation):
    """Stored beliefs about an object's physical properties, accumulated over
    prior manipulations."""


@ontological(iri=FLAN + "SensoryEventPrediction", aligns_to=DUL + "Description")
@dataclass(eq=False)
class SensoryEventPrediction(KnowledgeEntity):
    """A model-generated sensory consequence expected at a mechanical event."""

    modality: Optional[SensorySignalType] = prop(
        functional=True, min_count=1, iri=FLAN + "hasModality"
    )
    """The sensory channel in which the consequence is expected."""
    predicts_event: Optional[PhaseBoundaryEventType] = prop(
        functional=True, iri=FLAN + "predictsEvent"
    )
    """The expected event, mutually exclusive with :attr:`predicts_feature`."""
    predicts_feature: Optional[ConceptReference] = prop(
        functional=True, iri=FLAN + "predictsFeature"
    )
    """The expected signal feature, mutually exclusive with :attr:`predicts_event`."""
    caused_by_event: Optional[MechanicalEventType] = prop(
        functional=True, min_count=1, iri=FLAN + "causedByEvent"
    )
    """The mechanical event expected to produce the sensory consequence."""
    expected_timing: Optional[TemporalWindow] = prop(
        functional=True, min_count=1, iri=FLAN + "expectedTiming"
    )
    """The interval in which the consequence is expected."""
    expected_on_role: list[RoleSlot] = prop(
        min_count=1, default_factory=list, iri=FLAN + "expectedOnRole"
    )
    """The participant roles at which the consequence is expected."""
    expected_magnitude: Optional[MagnitudeRange] = prop(
        functional=True, iri=FLAN + "expectedMagnitude"
    )
    """The expected magnitude when the prediction is quantitative."""
    tolerance: Optional[MagnitudeRange] = prop(
        functional=True, iri=FLAN + "hasTolerance"
    )
    """The admissible quantitative deviation from the expected magnitude."""
    generated_by: Optional[ForwardModel] = prop(
        functional=True, min_count=1, iri=FLAN + "generatedBy"
    )
    """The forward model that produced the prediction."""


@ontological(iri=FLAN + "SensoryObservationType", aligns_to=DUL + "EventType")
@dataclass(eq=False)
class SensoryObservationType(KnowledgeEntity):
    """A type-level sensory result available for comparison with a prediction."""

    modality: Optional[SensorySignalType] = prop(
        functional=True, min_count=1, iri=FLAN + "hasModality"
    )
    """The sensory channel carrying the observation."""
    observes_event: Optional[PhaseBoundaryEventType] = prop(
        functional=True, iri=FLAN + "observesEvent"
    )
    """The observed event, mutually exclusive with :attr:`observes_feature`."""
    observes_feature: Optional[ConceptReference] = prop(
        functional=True, iri=FLAN + "observesFeature"
    )
    """The observed feature, mutually exclusive with :attr:`observes_event`."""
    observed_at: Optional[TemporalWindow] = prop(
        functional=True, min_count=1, iri=FLAN + "observedAt"
    )
    """The interval in which the observation occurs."""
    observed_on_role: list[RoleSlot] = prop(
        min_count=1, default_factory=list, iri=FLAN + "observedOnRole"
    )
    """The participant roles involved in the observation."""
    observed_magnitude: Optional[MagnitudeRange] = prop(
        functional=True, iri=FLAN + "observedMagnitude"
    )
    """The observed magnitude when the observation is quantitative."""


@ontological(iri=FLAN + "SignalTrajectory", aligns_to=DUL + "Description")
@dataclass(eq=False)
class SignalTrajectory(KnowledgeEntity):
    """A structured change in one sensory feature over a time interval."""

    signal_feature: Optional[ConceptReference] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "predictsFeature",
    )
    """The sensory feature whose evolution is predicted."""
    trend: Optional[TrajectoryTrend] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasTrajectoryTrend",
    )
    """The expected direction of feature change."""
    expected_range: Optional[MagnitudeRange] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "expectedMagnitude",
    )
    """The range traversed or maintained by the feature."""
    expected_timing: Optional[TemporalWindow] = prop(
        functional=True,
        iri=FLAN + "expectedTiming",
    )
    """The interval over which the trajectory is expected."""


@ontological(iri=FLAN + "ContinuousSignalPrediction", aligns_to=DUL + "Description")
@dataclass(eq=False)
class ContinuousSignalPrediction(KnowledgeEntity):
    """A model-generated sensory trajectory compared throughout a phase."""

    signal: Optional[SensorySignalType] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasSignal",
    )
    """The sensory channel carrying the predicted trajectory."""
    predicted_trajectory: Optional[SignalTrajectory] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "predictedTrajectory",
    )
    """The structured feature change expected during the phase."""
    tolerance: Optional[MagnitudeRange] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasTolerance",
    )
    """The admissible deviation from the predicted trajectory."""
    generated_by: Optional[ForwardModel] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "generatedBy",
    )
    """The forward model producing the prediction."""
    comparison_target: Optional[SensoryObservationType] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "comparesObservation",
    )
    """The observation type continuously compared with the prediction."""
    controlled_variable: Optional[Parameter] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "controlsVariable",
    )
    """The parameter adjusted from the comparison error."""
    predicted_for_phase: list[ActionPhase] = prop(
        min_count=1,
        default_factory=list,
        iri=FLAN + "predictedForPhase",
    )
    """The phases during which the comparison remains active."""


@ontological(iri=FLAN + "PredictionError", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class PredictionError(KnowledgeEntity):
    """A classified discrepancy between a prediction and an observation."""

    kind: Optional[PredictionErrorKind] = prop(
        functional=True, min_count=1, iri=FLAN + "hasErrorKind"
    )
    """The comparison dimension along which the prediction failed."""
    discrepancy: Optional[MagnitudeRange] = prop(
        functional=True, iri=FLAN + "hasDiscrepancy"
    )
    """The discrepancy magnitude when the error is quantitative."""


@ontological(iri=FLAN + "CorrectiveAction", aligns_to=SOMA + "PhysicalTask")
@dataclass(eq=False)
class CorrectiveAction(KnowledgeEntity):
    """A typed motor response to one or more prediction errors."""

    triggered_by_error: list[PredictionError] = prop(
        min_count=1, default_factory=list, iri=FLAN + "triggeredByError"
    )
    """The prediction errors that release the response."""
    targets_setpoint: list[MotorSetpoint] = prop(
        min_count=1, default_factory=list, iri=FLAN + "targetsSetpoint"
    )
    """The motor setpoints changed by the response."""
    direction: Optional[CorrectionDirection] = prop(
        functional=True, min_count=1, iri=FLAN + "hasCorrectionDirection"
    )
    """How the targeted setpoints are changed."""
    intended_effect: list[PhaseConditionType] = prop(
        min_count=1, default_factory=list, iri=FLAN + "hasIntendedEffect"
    )
    """The state the correction is intended to establish or preserve."""
    recovery_mode: Optional[RecoveryMode] = prop(
        functional=True, min_count=1, iri=FLAN + "hasRecoveryMode"
    )
    """How phase execution proceeds after applying the response."""


@ontological(iri=FLAN + "ModelUpdate", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class ModelUpdate(KnowledgeEntity):
    """A prediction-error-driven revision of represented estimates."""

    triggered_by_error: list[PredictionError] = prop(
        min_count=1, default_factory=list, iri=FLAN + "triggeredByError"
    )
    """The prediction errors supplying evidence for the revision."""
    updates_estimate: list[Estimate] = prop(
        min_count=1, default_factory=list, iri=FLAN + "updatesEstimate"
    )
    """The estimates changed by the revision."""
    updates_representation: Optional[EstimateRepresentation] = prop(
        functional=True, min_count=1, iri=FLAN + "updatesRepresentation"
    )
    """The estimate representation in which the content is changed."""
    direction: Optional[EstimateUpdateDirection] = prop(
        functional=True, min_count=1, iri=FLAN + "hasUpdateDirection"
    )
    """How the represented estimate changes."""


@ontological(iri=FLAN + "InitialStateEstimate", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class InitialStateEstimate(KnowledgeEntity):
    """The observation-grounded state supplied to the next action phase."""

    for_phase: Optional[ActionPhase] = prop(
        functional=True, min_count=1, iri=FLAN + "forPhase"
    )
    """The phase whose initial state is estimated."""
    based_on_observation: list[SensoryObservationType] = prop(
        min_count=1, default_factory=list, iri=FLAN + "basedOnObservation"
    )
    """The observations supporting the state estimate."""
    estimates_condition: list[PhaseConditionType] = prop(
        min_count=1, default_factory=list, iri=FLAN + "estimatesCondition"
    )
    """The conditions estimated to hold at phase entry."""


@ontological(iri=FLAN + "PredictionComparison", aligns_to=DUL + "Situation")
@dataclass(eq=False)
class PredictionComparison(KnowledgeEntity):
    """The operational comparison between one prediction and its observation."""

    compares_prediction: Optional[SensoryEventPrediction] = prop(
        functional=True, min_count=1, iri=FLAN + "comparesPrediction"
    )
    """The sensory prediction being evaluated."""
    compares_observation: Optional[SensoryObservationType] = prop(
        functional=True, iri=FLAN + "comparesObservation"
    )
    """The received observation, absent only for an absent-event error."""
    outcome: Optional[ComparisonOutcome] = prop(
        functional=True, min_count=1, iri=FLAN + "hasComparisonOutcome"
    )
    """Whether the observation matches the prediction."""
    confirms_subgoal: Optional[Subgoal] = prop(
        functional=True, iri=FLAN + "confirmsSubgoal"
    )
    """The subgoal confirmed by a successful comparison."""
    supplies_state_estimate: Optional[InitialStateEstimate] = prop(
        functional=True, iri=FLAN + "suppliesStateEstimate"
    )
    """The next-phase state supplied by a successful comparison."""
    permits_transition: list[PhaseTransition] = prop(
        default_factory=list, iri=FLAN + "permitsTransition"
    )
    """The phase transitions released by a successful comparison."""
    has_error: list[PredictionError] = prop(
        default_factory=list, iri=FLAN + "hasPredictionError"
    )
    """The errors classified by a mismatching comparison."""
    invokes_correction: list[CorrectiveAction] = prop(
        default_factory=list, iri=FLAN + "invokesCorrection"
    )
    """The corrective responses released by a mismatch."""
    produces_model_update: list[ModelUpdate] = prop(
        default_factory=list, iri=FLAN + "producesModelUpdate"
    )
    """The representation updates supported by a mismatch."""


@ontological(iri=FLAN + "SensorimotorControlPoint", aligns_to=DUL + "Relation")
@dataclass(eq=False)
class SensorimotorControlPoint(KnowledgeEntity):
    """A mechanical boundary with explicit predictions and comparison outcomes."""

    at_event_type: Optional[MechanicalEventType] = prop(
        functional=True, min_count=1, iri=FLAN + "atEventType"
    )
    """The mechanical event at which comparisons are performed."""
    has_prediction: list[SensoryEventPrediction] = prop(
        min_count=1, default_factory=list, iri=FLAN + "hasPrediction"
    )
    """The sensory consequences expected at the event."""
    has_comparison: list[PredictionComparison] = prop(
        min_count=1, default_factory=list, iri=FLAN + "hasComparison"
    )
    """The possible match and mismatch outcomes at the control point."""


# %% phases and structure


@ontological(
    iri=FLAN + "ActionPhase", aligns_to=SOMA + "PhysicalTask", disjoint_siblings=True
)
@dataclass(eq=False)
class ActionPhase(KnowledgeEntity):
    """A discrete sensorimotor building block of a manipulation task.

    A task concept, not an occurrence. Every phase has exactly one subgoal, one
    coupling regime and one within-phase regulation mode. Its termination is
    represented independently by one or more boundary event types.
    """

    aims_for: Optional[Subgoal] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "aimsFor",
        comment="The single subgoal that individuates this phase.",
    )
    coupling_regime: Optional[CouplingRegime] = prop(
        functional=True, min_count=1, iri=FLAN + "hasCouplingRegime"
    )
    regulation_mode: Optional[RegulationMode] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "hasRegulationMode",
    )
    """How commands are regulated within the phase, independent of its boundary."""
    initiated_by: list[PhaseBoundaryEventType] = prop(
        default_factory=list,
        iri=FLAN + "initiatedBy",
        comment="Optional: the first phase of a structure has none.",
    )
    terminated_by: list[PhaseBoundaryEventType] = prop(
        min_count=1,
        default_factory=list,
        iri=FLAN + "terminatedBy",
        comment="A phase with no boundary is not a phase.",
    )
    phase_involves: list[RoleSlot] = prop(
        min_count=1, default_factory=list, iri=FLAN + "phaseInvolves"
    )
    has_condition_requirement: list[ConditionRequirement] = prop(
        default_factory=list, iri=FLAN + "hasConditionRequirement"
    )
    monitors_signal: list[SensorySignalType] = prop(
        default_factory=list,
        iri=FLAN + "monitorsSignal",
        comment="Tracked throughout the phase rather than sampled at a "
        "boundary. A continuously modulated phase with nothing "
        "monitored has no control law.",
    )
    governed_by: list[InternalModel] = prop(
        default_factory=list, iri=FLAN + "governedBy"
    )


@ontological(iri=FLAN + "PhaseTransition", aligns_to=DUL + "Relation")
@dataclass(eq=False)
class PhaseTransition(KnowledgeEntity):
    """A gated edge of the phase state machine.

    Control passes from one phase to the next when a declared boundary event is
    registered. Mechanical boundaries preserve the predictive-control semantics;
    typed external boundaries support interruption without pretending it is a
    physical event.
    """

    from_phase: Optional[ActionPhase] = prop(
        functional=True, min_count=1, iri=FLAN + "fromPhase"
    )
    to_phase: Optional[ActionPhase] = prop(
        functional=True, min_count=1, iri=FLAN + "toPhase"
    )
    gated_by: list[PhaseBoundaryEventType] = prop(
        min_count=1, default_factory=list, iri=FLAN + "gatedBy"
    )
    permitted_by: list[PredictionComparison] = prop(
        default_factory=list, iri=FLAN + "permittedBy"
    )
    """Successful prediction comparisons required in addition to event gates."""


@ontological(
    iri=FLAN + "ActionPhaseStructure",
    aligns_to=DUL + "Workflow",
    display_name="ActionPhases",
)
@dataclass(eq=False)
class ActionPhaseStructure(KnowledgeEntity):
    """The top-level unit of Flanagan knowledge: a phase decomposition of one
    manipulation task.

    One instance is one hypothesis about how a kind of manipulation is
    organised. Delete this node and the hypothesis is gone -- which is what makes
    it cacheable and versionable as a unit.
    """

    decomposes_task: Optional[ConceptReference] = prop(
        functional=True,
        min_count=1,
        iri=FLAN + "decomposesTask",
        comment="The overall task this decomposes, as a reference into SOMA. "
        "Exactly one: this is what makes the hypothesis retrievable by "
        "action type.",
    )
    defines_phase: list[ActionPhase] = prop(
        min_count=1, default_factory=list, iri=FLAN + "definesPhase"
    )
    defines_role: list[RoleSlot] = prop(
        min_count=1, default_factory=list, iri=FLAN + "definesRole"
    )
    defines_event_type: list[PhaseBoundaryEventType] = prop(
        min_count=1, default_factory=list, iri=FLAN + "definesEventType"
    )
    defines_parameter: list[Parameter] = prop(
        default_factory=list, iri=FLAN + "definesParameter"
    )
    defines_prediction: list[SensoryEventPrediction] = prop(
        default_factory=list, iri=FLAN + "definesPrediction"
    )
    """The sensory predictions declared by the hypothesis."""
    defines_observation: list[SensoryObservationType] = prop(
        default_factory=list, iri=FLAN + "definesObservation"
    )
    """The sensory observation types declared by the hypothesis."""
    has_comparison: list[PredictionComparison] = prop(
        default_factory=list, iri=FLAN + "hasComparison"
    )
    """The prediction comparisons declared by the hypothesis."""
    defines_error: list[PredictionError] = prop(
        default_factory=list, iri=FLAN + "definesError"
    )
    """The prediction errors declared by the hypothesis."""
    defines_correction: list[CorrectiveAction] = prop(
        default_factory=list, iri=FLAN + "definesCorrection"
    )
    """The corrective responses declared by the hypothesis."""
    has_model_update: list[ModelUpdate] = prop(
        default_factory=list, iri=FLAN + "hasModelUpdate"
    )
    """The model updates declared by the hypothesis."""
    defines_initial_state_estimate: list[InitialStateEstimate] = prop(
        default_factory=list, iri=FLAN + "definesInitialStateEstimate"
    )
    """The next-phase state estimates declared by the hypothesis."""
    has_transition: list[PhaseTransition] = prop(
        min_count=1, default_factory=list, iri=FLAN + "hasTransition"
    )
    has_control_point: list[SensorimotorControlPoint] = prop(
        default_factory=list, iri=FLAN + "hasControlPoint"
    )
