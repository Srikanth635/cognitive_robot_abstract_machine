"""
The hub every perspective on an action attaches to: the frame of a kind of action, the slots its participants fill, what it is for, what it brings about and the kinds of choice it leaves open.

A perspective, such as the action frame, the image schemas or predictive control, describes the action in terms of the core's slots and parameter dimensions and never in terms of another perspective. The resolver grounds a slot in an individual of the world through a CRAM-AICOR role binding, and maps the slots and parameter dimensions to the fields of an action designator.

Each concept is linked to CRAM-AICOR and to DUL: frames are descriptions, slots are CRAM-AICOR role slots specialising its participant roles, an objective is a CRAM-AICOR telos, a consequence a CRAM-AICOR effect, and a parameter dimension a DUL parameter in CRAM-AICOR's action parameter space. Axioms are read under the open-world assumption.

Generated from ``action-core.ttl`` by ``python -m llm_reasoner.knowledge.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import ClassVar, List, Optional

from llm_reasoner.knowledge import ontologies
from llm_reasoner.knowledge.ontologies.properties import ObjectPropertyDescriptor
from llm_reasoner.knowledge.ontologies.terms import (
    ClassTerm,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    OpenValuePartition,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)

# %% value partitions


class ChangeDirection(OpenValuePartition):
    """
    The direction in which something changes: larger or smaller, such as a quantity an action changes, a choice a correction moves or an estimate evidence revises, or a direction of its own a module adds, such as moving a choice towards a target. A property taking a direction may be closed to the members it takes.
    """

    INCREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Increases"
    """
    Becomes larger.
    """

    DECREASES = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Decreases"
    """
    Becomes smaller, or grows more slowly.
    """

    TOWARDS_TARGET = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#TowardsTarget"
    """
    The chosen position or orientation is moved towards a slot or part of one.
    """

    AWAY_FROM_TARGET = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#AwayFromTarget"
    """
    The chosen position or orientation is moved away from a slot or part of one.
    """

    TO_ALTERNATIVE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/failures#ToAlternative"
    """
    Another of the available alternatives is chosen.
    """

    HOLD_SETPOINT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#HoldSetpoint"
    """
    Keep the target at its present value.
    """

    REVERSE_CHANGE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ReverseChange"
    """
    Command a change opposite to the current direction.
    """

    STOP_CHANGE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#StopChange"
    """
    Bring the commanded change to rest.
    """

    REPLACE_ESTIMATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ReplaceEstimate"
    """
    Change an unordered estimate or replace its current value.
    """

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            ChangeDirection.INCREASES: ontologies.ActionPerspectiveFile.ACTION_CORE,
            ChangeDirection.DECREASES: ontologies.ActionPerspectiveFile.ACTION_CORE,
            ChangeDirection.TOWARDS_TARGET: ontologies.ActionPerspectiveFile.FAILURES,
            ChangeDirection.AWAY_FROM_TARGET: ontologies.ActionPerspectiveFile.FAILURES,
            ChangeDirection.TO_ALTERNATIVE: ontologies.ActionPerspectiveFile.FAILURES,
            ChangeDirection.HOLD_SETPOINT: ontologies.PredictiveManipulationOntologyFile.CORE,
            ChangeDirection.REVERSE_CHANGE: ontologies.PredictiveManipulationOntologyFile.CORE,
            ChangeDirection.STOP_CHANGE: ontologies.PredictiveManipulationOntologyFile.CORE,
            ChangeDirection.REPLACE_ESTIMATE: ontologies.PredictiveManipulationOntologyFile.CORE,
        }[self]


class ConditionPolarity(StrEnum):
    """
    Whether a condition holds or does not hold. The property relating a condition to its polarity says whether that is demanded, such as a requirement of a phase or a frame, or observed, such as the condition a failure is noticed by.
    """

    HOLDS = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Holds"
    )
    """
    The condition holds.
    """

    DOES_NOT_HOLD = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#DoesNotHold"
    """
    The condition does not hold.
    """


class ConditionTiming(OpenValuePartition):
    """
    When, relative to a phase or an action, a condition or a relation is demanded.
    """

    AT_ENTRY = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#AtEntry"
    """
    Must hold for the phase or the action to begin.
    """

    THROUGHOUT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Throughout"
    """
    Must hold at every instant of the phase or the action.
    """

    AT_EXIT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#AtExit"
    """
    Is brought about by completing the phase or the action.
    """

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            ConditionTiming.AT_ENTRY: ontologies.ActionPerspectiveFile.ACTION_CORE,
            ConditionTiming.THROUGHOUT: ontologies.ActionPerspectiveFile.ACTION_CORE,
            ConditionTiming.AT_EXIT: ontologies.ActionPerspectiveFile.ACTION_CORE,
        }[self]


class MeasurementUnit(OpenValuePartition):
    """
    A unit measuring exactly one physical dimension. Neither CRAM-AICOR nor DUL defines units.
    """

    KILOGRAM = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Kilogram"
    """
    Kilogram, a unit of mass.
    """

    GRAM = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Gram"
    )
    """
    Gram, a unit of mass.
    """

    KILOGRAM_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#KilogramPerSecond"
    """
    Kilogram per second, a unit of mass flow rate.
    """

    NEWTON = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Newton"
    """
    Newton, a unit of force.
    """

    NEWTON_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#NewtonPerSecond"
    """
    Newton per second, a unit of force rate.
    """

    NEWTON_METRE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#NewtonMetre"
    """
    Newton metre, a unit of torque.
    """

    METRE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Metre"
    )
    """
    Metre, a unit of length.
    """

    CENTIMETRE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Centimetre"
    """
    Centimetre, a unit of length.
    """

    MILLIMETRE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Millimetre"
    """
    Millimetre, a unit of length.
    """

    DEGREE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Degree"
    """
    Degree, a unit of angle.
    """

    DEGREE_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#DegreePerSecond"
    """
    Degree per second, a unit of angular rate.
    """

    LITRE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Litre"
    )
    """
    Litre, a unit of volume.
    """

    MILLILITRE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Millilitre"
    """
    Millilitre, a unit of volume.
    """

    MILLILITRE_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#MillilitrePerSecond"
    """
    Millilitre per second, a unit of volumetric flow rate.
    """

    CENTIPOISE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Centipoise"
    """
    Centipoise, a unit of dynamic viscosity.
    """

    HERTZ = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Hertz"
    )
    """
    Hertz, a unit of frequency.
    """

    SECOND = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Second"
    """
    Second, a unit of time.
    """

    MILLISECOND = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Millisecond"
    """
    Millisecond, a unit of time.
    """

    UNITLESS = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Unitless"
    """
    The unit of a dimensionless ratio or bounded fraction.
    """

    @property
    def measures_dimension(self) -> PhysicalDimension:
        """
        The single physical dimension a unit measures.
        """
        return {
            MeasurementUnit.KILOGRAM: PhysicalDimension.MASS,
            MeasurementUnit.GRAM: PhysicalDimension.MASS,
            MeasurementUnit.KILOGRAM_PER_SECOND: PhysicalDimension.MASS_FLOW_RATE,
            MeasurementUnit.NEWTON: PhysicalDimension.FORCE,
            MeasurementUnit.NEWTON_PER_SECOND: PhysicalDimension.FORCE_RATE,
            MeasurementUnit.NEWTON_METRE: PhysicalDimension.TORQUE,
            MeasurementUnit.METRE: PhysicalDimension.LENGTH,
            MeasurementUnit.CENTIMETRE: PhysicalDimension.LENGTH,
            MeasurementUnit.MILLIMETRE: PhysicalDimension.LENGTH,
            MeasurementUnit.DEGREE: PhysicalDimension.ANGLE,
            MeasurementUnit.DEGREE_PER_SECOND: PhysicalDimension.ANGULAR_RATE,
            MeasurementUnit.LITRE: PhysicalDimension.VOLUME,
            MeasurementUnit.MILLILITRE: PhysicalDimension.VOLUME,
            MeasurementUnit.MILLILITRE_PER_SECOND: PhysicalDimension.VOLUMETRIC_FLOW_RATE,
            MeasurementUnit.CENTIPOISE: PhysicalDimension.DYNAMIC_VISCOSITY,
            MeasurementUnit.HERTZ: PhysicalDimension.FREQUENCY,
            MeasurementUnit.SECOND: PhysicalDimension.TIME,
            MeasurementUnit.MILLISECOND: PhysicalDimension.TIME,
            MeasurementUnit.UNITLESS: PhysicalDimension.DIMENSIONLESS,
        }[self]

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            MeasurementUnit.KILOGRAM: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.GRAM: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.KILOGRAM_PER_SECOND: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.NEWTON: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.NEWTON_PER_SECOND: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.NEWTON_METRE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.METRE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.CENTIMETRE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.MILLIMETRE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.DEGREE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.DEGREE_PER_SECOND: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.LITRE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.MILLILITRE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.MILLILITRE_PER_SECOND: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.CENTIPOISE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.HERTZ: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.SECOND: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.MILLISECOND: ontologies.ActionPerspectiveFile.ACTION_CORE,
            MeasurementUnit.UNITLESS: ontologies.ActionPerspectiveFile.ACTION_CORE,
        }[self]


class PhysicalDimension(OpenValuePartition):
    """
    A dimensional space of physical quantities, which in DUL is a maximal region.
    """

    MASS = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Mass"
    )
    """
    Quantity of matter. Distinct from weight, which is a force.
    """

    MASS_FLOW_RATE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#MassFlowRate"
    """
    Mass transferred per unit time.
    """

    FORCE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Force"
    )
    """
    Mechanical force, including weight force.
    """

    FORCE_RATE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ForceRate"
    """
    Rate of force change.
    """

    TORQUE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Torque"
    """
    Moment of force.
    """

    LENGTH = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Length"
    """
    Spatial distance or position component.
    """

    ANGLE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Angle"
    )
    """
    Planar or rotational angle.
    """

    ANGULAR_RATE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#AngularRate"
    """
    Rate of angular change.
    """

    VOLUME = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Volume"
    """
    Three-dimensional capacity.
    """

    VOLUMETRIC_FLOW_RATE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#VolumetricFlowRate"
    """
    Volume transferred per unit time.
    """

    DYNAMIC_VISCOSITY = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#DynamicViscosity"
    """
    Resistance of a fluid to shear flow.
    """

    FREQUENCY = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Frequency"
    """
    Repetitions of a periodic signal per unit time.
    """

    TIME = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Time"
    )
    """
    Temporal duration or offset.
    """

    DIMENSIONLESS = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Dimensionless"
    """
    A ratio or fraction without a physical unit.
    """

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            PhysicalDimension.MASS: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.MASS_FLOW_RATE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.FORCE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.FORCE_RATE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.TORQUE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.LENGTH: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.ANGLE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.ANGULAR_RATE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.VOLUME: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.VOLUMETRIC_FLOW_RATE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.DYNAMIC_VISCOSITY: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.FREQUENCY: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.TIME: ontologies.ActionPerspectiveFile.ACTION_CORE,
            PhysicalDimension.DIMENSIONLESS: ontologies.ActionPerspectiveFile.ACTION_CORE,
        }[self]


class QuantityKind(OpenValuePartition):
    """
    What a quantity about the filler of a slot is: how much, how fast, where, which way it faces, how far, how long, how strongly it acts, a property of its material, a ratio, its shape, or which of several.
    """

    AMOUNT = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Amount"
    """
    How much of something, such as the volume poured.
    """

    RATE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Rate"
    )
    """
    How fast something changes, such as the flow or the tilting.
    """

    POSITION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Position"
    """
    Where something is, such as where the lip is over the goal.
    """

    ORIENTATION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Orientation"
    """
    Which way something faces, such as how far the source is tilted.
    """

    DISTANCE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Distance"
    """
    How far apart two things are, such as the height of the lip above the goal.
    """

    DURATION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Duration"
    """
    How long something lasts.
    """

    SELECTION = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Selection"
    """
    Which of several alternatives is taken, such as the hand that holds the source.
    """

    MAGNITUDE = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Magnitude"
    """
    How strongly one filler acts on another, such as the grip force on the source or the torque about the grasp.
    """

    MATERIAL_PROPERTY = "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#MaterialProperty"
    """
    A property of the material of a filler or of its surface, such as the viscosity of the substance, the friction of the grasp or how the substance flows.
    """

    RATIO = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Ratio"
    )
    """
    How large one quantity is against another, such as the grip force against the load force or the filled fraction of the goal.
    """

    SHAPE = (
        "http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Shape"
    )
    """
    What form a filler or its surface has, such as the geometry of the grasp surfaces.
    """

    @property
    def measured_in(self) -> tuple[PhysicalDimension, ...]:
        """
        A physical dimension a kind of quantity may be measured in, such as an angle for an orientation; a kind measured in none, such as a selection or a shape, constrains none.
        """
        return {
            QuantityKind.AMOUNT: (
                PhysicalDimension.MASS,
                PhysicalDimension.VOLUME,
            ),
            QuantityKind.RATE: (
                PhysicalDimension.MASS_FLOW_RATE,
                PhysicalDimension.FORCE_RATE,
                PhysicalDimension.ANGULAR_RATE,
                PhysicalDimension.VOLUMETRIC_FLOW_RATE,
                PhysicalDimension.FREQUENCY,
            ),
            QuantityKind.POSITION: (PhysicalDimension.LENGTH,),
            QuantityKind.ORIENTATION: (PhysicalDimension.ANGLE,),
            QuantityKind.DISTANCE: (PhysicalDimension.LENGTH,),
            QuantityKind.DURATION: (PhysicalDimension.TIME,),
            QuantityKind.SELECTION: (),
            QuantityKind.MAGNITUDE: (
                PhysicalDimension.FORCE,
                PhysicalDimension.TORQUE,
            ),
            QuantityKind.MATERIAL_PROPERTY: (
                PhysicalDimension.DYNAMIC_VISCOSITY,
                PhysicalDimension.DIMENSIONLESS,
            ),
            QuantityKind.RATIO: (PhysicalDimension.DIMENSIONLESS,),
            QuantityKind.SHAPE: (),
        }[self]

    @property
    def declaring_module(self) -> ontologies.OntologyModuleFile:
        return {
            QuantityKind.AMOUNT: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.RATE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.POSITION: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.ORIENTATION: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.DISTANCE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.DURATION: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.SELECTION: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.MAGNITUDE: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.MATERIAL_PROPERTY: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.RATIO: ontologies.ActionPerspectiveFile.ACTION_CORE,
            QuantityKind.SHAPE: ontologies.ActionPerspectiveFile.ACTION_CORE,
        }[self]


# %% classes


@dataclass(eq=False)
class ActionFailure(DescriptionEntity):
    """
    A way an action can go wrong, which a perspective describes and an event or an observation can signal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFailure",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#FailureDescription",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ActionFrame(DescriptionEntity):
    """
    The frame of a kind of action: the slots its participants fill, what it is for, what it brings about and the kinds of choice it leaves open. Every perspective on the action refers to the slots its frame defines.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ActionFrame",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#statesObjective",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    brings_about: List[Consequence] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#bringsAbout",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Consequence",
        ).as_dict(),
    )
    """
    A change the action makes to one of its participants.
    """

    defines_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#definesSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot of the frame, which a participant of the action, or a part of one, fills.
    """

    frames_action: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#framesAction",
            functional=True,
            linked_range="http://aicor.knowledge/l2-ontology.owl#ManipulationActionCategory",
        ).as_dict(),
    )
    """
    The kind of action the frame describes, as the IRI of a CRAM-AICOR action category, such as aicor:Pouring.
    """

    leaves_open: List[ParameterDimension] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#leavesOpen",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        ).as_dict(),
    )
    """
    A kind of choice the frame does not settle, which whoever performs the action makes.
    """

    states_objective: List[Objective] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#statesObjective",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective",
        ).as_dict(),
    )
    """
    What the action is for.
    """


@dataclass(eq=False)
class FrameSlot(DescriptionEntity):
    """
    A place in a frame that something taking part in the action fills; a CRAM-AICOR role binding binds an individual of the world to it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#RoleSlot",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ParticipantSlot(FrameSlot):
    """
    A slot that a participant of the action fills as a whole, such as the source of a pour.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class AgentSlot(ParticipantSlot):
    """
    The slot of the one who performs the action.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#AgentSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#AgentRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Consequence(DescriptionEntity):
    """
    A change an action makes to one of its participants, intended or not, such as the source of a pour holding less, which CRAM-AICOR calls an effect.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Consequence",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Effect",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#affectsSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    affects_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#affectsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot whose filler the consequence changes.
    """

    results_in_state: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#resultsInState",
            functional=True,
            linked_range="http://aicor.knowledge/l2-ontology.owl#DiscreteMode",
        ).as_dict(),
    )
    """
    The state the filler is in afterwards, as the IRI of a CRAM-AICOR state type, such as aicor:ContainedState.
    """


@dataclass(eq=False)
class GoalSlot(ParticipantSlot):
    """
    The slot of where what the action moves ends up, such as the container poured into.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#GoalSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#GoalRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InstrumentSlot(ParticipantSlot):
    """
    The slot of a tool the agent acts with on something else.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#InstrumentSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#InstrumentRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ManipulationActionFrame(ActionFrame):
    """
    The frame of an action that manipulates objects, which CRAM-AICOR calls a manipulation frame; an action that manipulates nothing, such as moving the robot's base, has a frame that is not one.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ManipulationActionFrame",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#ManipulationFrame",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Objective(DescriptionEntity):
    """
    What an action is for: a state of its participants it aims at, which CRAM-AICOR calls a telos.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#Objective",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#Telos",),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#concernsSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    aims_at_state: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aimsAtState",
            functional=True,
            linked_range="http://aicor.knowledge/l2-ontology.owl#DiscreteMode",
        ).as_dict(),
    )
    """
    The state the objective aims at, as the IRI of a CRAM-AICOR state type, such as aicor:AccumulatedState.
    """

    concerns_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#concernsSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    A slot whose filler the objective is about.
    """


@dataclass(eq=False)
class SlotQuantity(DescriptionEntity):
    """
    A kind of quantity about the filler of a slot, possibly measured from the filler of another slot or acting between it and the fillers of others, such as the tilt of the source, the height of the lip above the goal or the grip force on the source; it carries no value of its own. What it is for is said by its class: a choice the action leaves open, a quantity deciding whether a relation holds, one a failure is noticed by, or a parameter of a phase, which holds a value.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SlotQuantity",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Concept",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aboutSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#QuantityKind"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    about_slot: Optional[FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#aboutSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the quantity is about, such as the source whose orientation is chosen.
    """

    has_physical_dimension: Optional[PhysicalDimension] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasPhysicalDimension",
            functional=True,
        ).as_dict(),
    )
    """
    The physical dimension a quantity is measured in, which one of its kind's dimensions is and the unit of its value measures.
    """

    has_quantity_kind: Optional[QuantityKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasQuantityKind",
            functional=True,
        ).as_dict(),
    )
    """
    What kind of quantity something about an action is about, such as the quantity a choice sets, a condition constrains or a change changes.
    """

    quantity_involves_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#quantityInvolvesSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    Another slot whose filler the quantity acts between the first filler and, such as the effector a grip force on the source is exerted by, or the goal a transferred amount ends up in; unlike relative_to_slot, it is no reference the quantity is measured from.
    """

    relative_to_slot: Optional[FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#relativeToSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the quantity is measured from, such as the goal a position above which is chosen, or the opening the lip ends beside; absent for a quantity measured from nothing in particular.
    """


@dataclass(eq=False)
class ParameterDimension(SlotQuantity):
    """
    A kind of choice an action leaves open, independent of any designator: how much, how fast, where relative to another participant, which way something faces, how far or how long, or which of several. A designator's field is one way of making the choice.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParameterDimension",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ActionParameterSpace",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Parameter",
        ),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PartSlot(FrameSlot):
    """
    A slot that a part of a participant fills, which the action involves on its own, such as the lip of the source of a pour or the opening of its goal.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PartSlot",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#isPartOfSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    is_part_of_slot: Optional[ParticipantSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#isPartOfSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ParticipantSlot",
        ).as_dict(),
    )
    """
    The participant slot whose filler the part belongs to.
    """


@dataclass(eq=False)
class PatientSlot(ParticipantSlot):
    """
    The slot of what the action acts on and changes.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#PatientSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#PatientRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SlotConditionType(DescriptionEntity):
    """
    A kind of condition a world can check about the filler of a slot, possibly in relation to the fillers of other slots, such as having a part, affording something or being in a state, rather than membership of a class. Whether it holds, or must hold, is said by what states or uses it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SlotConditionType",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionOnSlot",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    condition_involves_slot: List[FrameSlot] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionInvolvesSlot",
            functional=False,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    Another slot whose filler the condition relates the first filler to, such as the agent in the condition that the source is held, or the opening in the condition that the lip is over it.
    """

    condition_on_slot: Optional[FrameSlot] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionOnSlot",
            functional=True,
            range="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#FrameSlot",
        ).as_dict(),
    )
    """
    The slot whose filler the condition is about.
    """


@dataclass(eq=False)
class SlotCondition(SlotConditionType):
    """
    A slot condition type stated as holding or not holding, such as the source not being empty. Whether it is required or observed is said by the property relating it to what uses it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SlotCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasPolarity",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#ConditionPolarity"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_polarity: Optional[ConditionPolarity] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#hasPolarity",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the condition holds or does not hold, or must or must not.
    """


@dataclass(eq=False)
class SourceSlot(ParticipantSlot):
    """
    The slot of where something the action moves comes from, such as the container poured from.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SourceSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SourceRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class StateCondition(SlotCondition):
    """
    A condition that the filler of a slot, possibly in relation to the filler of another slot, is in a given discrete state, such as the agent grasping the source or the substance being contained in the goal; with the polarity that it does not hold, that it is not in that state, such as the source being empty: the substance not contained in it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#StateCondition",
        linked_terms=(),
        evidence_status="",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionHasState",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    condition_has_state: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#conditionHasState",
            functional=False,
            linked_range="http://aicor.knowledge/l2-ontology.owl#DiscreteMode",
        ).as_dict(),
    )
    """
    The state the filler is in when the condition holds, as the IRI of a CRAM-AICOR state type, such as aicor:GraspedState or aicor:FlowingState.
    """


@dataclass(eq=False)
class SubstanceSlot(ParticipantSlot):
    """
    The slot of the material the action moves, as distinct from what holds it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/action-perspectives/action-core#SubstanceSlot",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#SubstanceRole",),
        evidence_status="",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class AboutSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the quantity is about, such as the source whose orientation is chosen.
    """


@dataclass
class AffectsSlot(ObjectPropertyDescriptor):
    """
    A slot whose filler the consequence changes.
    """


@dataclass
class BringsAbout(ObjectPropertyDescriptor):
    """
    A change the action makes to one of its participants.
    """


@dataclass
class ConcernsSlot(ObjectPropertyDescriptor):
    """
    A slot whose filler the objective is about.
    """


@dataclass
class ConditionInvolvesSlot(ObjectPropertyDescriptor):
    """
    Another slot whose filler the condition relates the first filler to, such as the agent in the condition that the source is held, or the opening in the condition that the lip is over it.
    """


@dataclass
class ConditionOnSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the condition is about.
    """


@dataclass
class DefinesSlot(ObjectPropertyDescriptor):
    """
    A slot of the frame, which a participant of the action, or a part of one, fills.
    """


@dataclass
class IsPartOfSlot(ObjectPropertyDescriptor):
    """
    The participant slot whose filler the part belongs to.
    """


@dataclass
class LeavesOpen(ObjectPropertyDescriptor):
    """
    A kind of choice the frame does not settle, which whoever performs the action makes.
    """


@dataclass
class QuantityInvolvesSlot(ObjectPropertyDescriptor):
    """
    Another slot whose filler the quantity acts between the first filler and, such as the effector a grip force on the source is exerted by, or the goal a transferred amount ends up in; unlike relative_to_slot, it is no reference the quantity is measured from.
    """


@dataclass
class RelativeToSlot(ObjectPropertyDescriptor):
    """
    The slot whose filler the quantity is measured from, such as the goal a position above which is chosen, or the opening the lip ends beside; absent for a quantity measured from nothing in particular.
    """


@dataclass
class SignalsFailure(ObjectPropertyDescriptor):
    """
    A way the action goes wrong that the event or the observation signals.
    """


@dataclass
class StatesObjective(ObjectPropertyDescriptor):
    """
    What the action is for.
    """


# %% descriptor attachments

SlotQuantity.about_slot = AboutSlot(SlotQuantity, "about_slot")
Consequence.affects_slot = AffectsSlot(Consequence, "affects_slot")
ActionFrame.brings_about = BringsAbout(ActionFrame, "brings_about")
Objective.concerns_slot = ConcernsSlot(Objective, "concerns_slot")
SlotConditionType.condition_involves_slot = ConditionInvolvesSlot(
    SlotConditionType, "condition_involves_slot"
)
SlotConditionType.condition_on_slot = ConditionOnSlot(
    SlotConditionType, "condition_on_slot"
)
ActionFrame.defines_slot = DefinesSlot(ActionFrame, "defines_slot")
PartSlot.is_part_of_slot = IsPartOfSlot(PartSlot, "is_part_of_slot")
ActionFrame.leaves_open = LeavesOpen(ActionFrame, "leaves_open")
SlotQuantity.quantity_involves_slot = QuantityInvolvesSlot(
    SlotQuantity, "quantity_involves_slot"
)
SlotQuantity.relative_to_slot = RelativeToSlot(SlotQuantity, "relative_to_slot")
ActionFrame.states_objective = StatesObjective(ActionFrame, "states_objective")
