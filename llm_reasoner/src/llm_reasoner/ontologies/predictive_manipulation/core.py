"""
The action-neutral core of a microworld for predictive object manipulation, drawn from "Control strategies in object manipulation tasks" (2006), "Coding and use of tactile signals from the fingertips in object manipulation tasks" (2009), "Sensory control of object manipulation" (2009) and "Motor prediction" (2001).

A task is a sensory plan: a sequence of subgoals, each signalled by predicted sensory events. Action-phase controllers execute the phases; at control points predicted and actual signals are compared, and a mismatch launches a corrective programme and updates the represented object properties.

Classes classify description-level individuals, the knowledge a language model generates; executions are described with CRAM-AICOR occurrences, occasions and role bindings. Each concept is linked to CRAM-AICOR and to DUL in the same DUL category, and CRAM-AICOR's dul_* copies are never equated with DUL. Axioms are read under the open-world assumption.

Generated from ``core.ttl`` by ``python -m llm_reasoner.ontology_translation``. Regenerate rather than edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from krrood.ontomatic.property_descriptor.property_descriptor import PropertyDescriptor
from typing_extensions import ClassVar, List, Optional

from llm_reasoner.ontologies.terms import (
    ClassTerm,
    ComplementOf,
    ConceptReference,
    DescriptionEntity,
    EnumeratedClass,
    IntersectionOf,
    NamedClass,
    PropertyRestriction,
    PropertyTerm,
    Quantifier,
    UnionOf,
)

# %% value partitions


class ComparisonOutcome(StrEnum):
    """
    Whether an observation agrees with its sensory prediction.
    """

    MATCH = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Match"
    )
    """
    The observation supports the prediction within its tolerances.
    """

    MISMATCH = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Mismatch"
    )
    """
    The observation differs from the prediction in a classified way.
    """


class ConditionMode(StrEnum):
    """
    When, relative to a phase, a condition is demanded.
    """

    AT_ENTRY = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AtEntry"
    )
    """
    Must hold for the phase to begin.
    """

    THROUGHOUT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Throughout"
    """
    Must hold at every instant of the phase.
    """

    AT_EXIT = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AtExit"
    )
    """
    Is brought about by completing the phase.
    """


class ConditionPolarity(StrEnum):
    """
    Whether a condition is demanded to hold or to be absent.
    """

    MUST_HOLD = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MustHold"
    )
    """
    The condition is demanded to be true.
    """

    MUST_NOT_HOLD = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MustNotHold"
    """
    The condition is demanded to be false.
    """


class CorrectionDirection(StrEnum):
    """
    How a corrective response changes its target setpoint.
    """

    RAISE_SETPOINT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RaiseSetpoint"
    """
    Raise the target setpoint.
    """

    LOWER_SETPOINT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LowerSetpoint"
    """
    Lower the target setpoint, or slow its rate of increase.
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


class CouplingRegime(StrEnum):
    """
    Whether the effector and the manipulandum form a single kinematic unit.
    """

    UNCOUPLED = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Uncoupled"
    """
    The effector moves without being mechanically bound to the object.
    """

    TRANSITIONAL = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Transitional"
    """
    Contact or force conditions are changing but the object does not translate.
    """

    COUPLED = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Coupled"
    )
    """
    The effector and the object move together as one kinematic unit.
    """


class EstimateUpdateDirection(StrEnum):
    """
    How evidence changes a represented estimate.
    """

    RAISE_ESTIMATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RaiseEstimate"
    """
    Revise the estimate upward.
    """

    LOWER_ESTIMATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LowerEstimate"
    """
    Revise the estimate downward.
    """

    REPLACE_ESTIMATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ReplaceEstimate"
    """
    Change an unordered estimate or replace its current value.
    """


class EvidenceStatus(StrEnum):
    """
    The degree to which the sources support a represented claim.
    """

    EMPIRICAL_FINDING = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding"
    """
    The reviewed experiments directly report the behaviour or dissociation.
    """

    SUPPORTED_INTERPRETATION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation"
    """
    The authors infer a control organisation or representation from converging findings.
    """

    AUTHOR_HYPOTHESIS = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis"
    """
    The authors propose a mechanism or functional role that is not yet established.
    """

    OPEN_QUESTION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#OpenQuestion"
    """
    The source states that the mechanism, representation or relationship remains unresolved.
    """

    ENGINEERING_EXTENSION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension"
    """
    Added for robotic knowledge or a task extension; the sources do not make this claim.
    """


class ForceKind(StrEnum):
    """
    The physical kind of a force acting in a manipulation.
    """

    WEIGHT = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Weight"
    )
    """
    Gravity acting on a participant's mass.
    """

    INERTIA = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Inertia"
    )
    """
    Resistance of a participant's mass to acceleration.
    """

    LOAD_FORCE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LoadForce"
    """
    The force tangential to the grasp surfaces that lifts, carries or moves the object.
    """

    GRIP_FORCE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#GripForce"
    """
    The force normal to the grasp surfaces that the digits apply.
    """

    FRICTION = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Friction"
    )
    """
    Tangential resistance at a contact, at most the grip force times the friction coefficient.
    """

    SUPPORT_REACTION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportReaction"
    """
    The force with which a supporting surface bears a weight.
    """

    CONTAINMENT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Containment"
    """
    The force with which container walls hold a substance in.
    """

    MATERIAL_STRENGTH = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MaterialStrength"
    """
    The internal resistance of an object to deformation or rupture.
    """


class MeasurementUnit(StrEnum):
    """
    A unit measuring exactly one physical dimension. Neither CRAM-AICOR nor DUL defines units.
    """

    KILOGRAM = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Kilogram"
    )
    """
    Kilogram, a unit of mass.
    """

    KILOGRAM_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#KilogramPerSecond"
    """
    Kilogram per second, a unit of mass flow rate.
    """

    NEWTON = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Newton"
    )
    """
    Newton, a unit of force.
    """

    NEWTON_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#NewtonPerSecond"
    """
    Newton per second, a unit of force rate.
    """

    NEWTON_METRE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#NewtonMetre"
    """
    Newton metre, a unit of torque.
    """

    METRE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Metre"
    )
    """
    Metre, a unit of length.
    """

    DEGREE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Degree"
    )
    """
    Degree, a unit of angle.
    """

    DEGREE_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DegreePerSecond"
    """
    Degree per second, a unit of angular rate.
    """

    MILLILITRE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Millilitre"
    """
    Millilitre, a unit of volume.
    """

    MILLILITRE_PER_SECOND = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MillilitrePerSecond"
    """
    Millilitre per second, a unit of volumetric flow rate.
    """

    CENTIPOISE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Centipoise"
    """
    Centipoise, a unit of dynamic viscosity.
    """

    HERTZ = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Hertz"
    )
    """
    Hertz, a unit of frequency.
    """

    SECOND = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Second"
    )
    """
    Second, a unit of time.
    """

    MILLISECOND = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Millisecond"
    """
    Millisecond, a unit of time.
    """

    RATIO = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Ratio"
    )
    """
    Unit marker for dimensionless ratios and bounded fractions.
    """

    @property
    def measures_dimension(self) -> PhysicalDimension:
        """
        The single physical dimension a unit measures.
        """
        return {
            MeasurementUnit.KILOGRAM: PhysicalDimension.MASS,
            MeasurementUnit.KILOGRAM_PER_SECOND: PhysicalDimension.MASS_FLOW_RATE,
            MeasurementUnit.NEWTON: PhysicalDimension.FORCE,
            MeasurementUnit.NEWTON_PER_SECOND: PhysicalDimension.FORCE_RATE,
            MeasurementUnit.NEWTON_METRE: PhysicalDimension.TORQUE,
            MeasurementUnit.METRE: PhysicalDimension.LENGTH,
            MeasurementUnit.DEGREE: PhysicalDimension.ANGLE,
            MeasurementUnit.DEGREE_PER_SECOND: PhysicalDimension.ANGULAR_RATE,
            MeasurementUnit.MILLILITRE: PhysicalDimension.VOLUME,
            MeasurementUnit.MILLILITRE_PER_SECOND: PhysicalDimension.VOLUMETRIC_FLOW_RATE,
            MeasurementUnit.CENTIPOISE: PhysicalDimension.DYNAMIC_VISCOSITY,
            MeasurementUnit.HERTZ: PhysicalDimension.FREQUENCY,
            MeasurementUnit.SECOND: PhysicalDimension.TIME,
            MeasurementUnit.MILLISECOND: PhysicalDimension.TIME,
            MeasurementUnit.RATIO: PhysicalDimension.DIMENSIONLESS,
        }[self]


class ParameterValueForm(StrEnum):
    """
    Whether a parameter holds a magnitude or a named category.
    """

    QUANTITATIVE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Quantitative"
    """
    The parameter holds a magnitude range with a unit.
    """

    CATEGORICAL = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Categorical"
    """
    The parameter holds a categorical value.
    """


class PhysicalDimension(StrEnum):
    """
    A dimensional space of physical quantities, which in DUL is a maximal region.
    """

    MASS = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Mass"
    """
    Quantity of matter. Distinct from weight, which is a force.
    """

    MASS_FLOW_RATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassFlowRate"
    """
    Mass transferred per unit time.
    """

    FORCE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Force"
    )
    """
    Mechanical force, including weight force.
    """

    FORCE_RATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceRate"
    """
    Rate of force change.
    """

    TORQUE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Torque"
    )
    """
    Moment of force.
    """

    LENGTH = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Length"
    )
    """
    Spatial distance or position component.
    """

    ANGLE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Angle"
    )
    """
    Planar or rotational angle.
    """

    ANGULAR_RATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AngularRate"
    """
    Rate of angular change.
    """

    VOLUME = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Volume"
    )
    """
    Three-dimensional capacity.
    """

    VOLUMETRIC_FLOW_RATE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumetricFlowRate"
    """
    Volume transferred per unit time.
    """

    DYNAMIC_VISCOSITY = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DynamicViscosity"
    """
    Resistance of a fluid to shear flow.
    """

    FREQUENCY = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Frequency"
    """
    Repetitions of a periodic signal per unit time.
    """

    TIME = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Time"
    """
    Temporal duration or offset.
    """

    DIMENSIONLESS = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Dimensionless"
    """
    A ratio or fraction without a physical unit.
    """


class PredictionErrorKind(StrEnum):
    """
    The dimension along which an observation violates a prediction.
    """

    ABSENT_EVENT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AbsentEvent"
    """
    The predicted event or feature was not observed.
    """

    UNEXPECTED_EVENT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#UnexpectedEvent"
    """
    An event or feature outside the prediction was observed.
    """

    EARLY_EVENT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EarlyEvent"
    """
    The predicted event or feature occurred before its timing window.
    """

    LATE_EVENT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LateEvent"
    """
    The predicted event or feature occurred after its timing window.
    """

    MAGNITUDE_MISMATCH = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MagnitudeMismatch"
    """
    The observed magnitude fell outside the prediction and its tolerance.
    """

    DIRECTION_MISMATCH = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DirectionMismatch"
    """
    The observed change proceeded in the opposite direction.
    """

    SPATIAL_MISMATCH = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SpatialMismatch"
    """
    The observation involved a different location or participant.
    """


class PredictionSource(StrEnum):
    """
    Where an estimate used to parameterise a controller in advance comes from.
    """

    OBJECT_SPECIFIC_MEMORY = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ObjectSpecificMemory"
    """
    Sensorimotor memory of earlier interactions with this particular object.
    """

    CATEGORY_SIZE_WEIGHT_ASSOCIATION = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CategorySizeWeightAssociation"
    """
    A learned size-weight relation within a category of familiar objects.
    """

    VISUAL_IDENTITY_CUE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VisualIdentityCue"
    """
    Visual recognition of a familiar object, used to retrieve its remembered properties.
    """

    VISUAL_GEOMETRIC_CUE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VisualGeometricCue"
    """
    Visually seen size and shape, used to predict mass distribution, stable grasp sites and force directions.
    """

    LEARNED_ARBITRARY_CUE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LearnedArbitraryCue"
    """
    A visual or auditory cue, such as a colour, learned to signal a mechanical property.
    """

    RECENT_HAPTIC_EXPERIENCE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RecentHapticExperience"
    """
    Memory of recent contacts with the same or similar surfaces, the only advance source for friction.
    """

    TACTILE_CONTACT_RESPONSE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TactileContactResponse"
    """
    Tactile signals at initial contact, which reveal friction and override visual shape predictions.
    """


class RecoveryMode(StrEnum):
    """
    How execution proceeds when a corrective response is released.
    """

    CONTINUE_PHASE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContinuePhase"
    """
    Apply the correction while the running controller continues.
    """

    ABORT_AND_CORRECT = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AbortAndCorrect"
    """
    Abort the running controller and execute the corrective programme in its place before the task proceeds.
    """

    REPEAT_PHASE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RepeatPhase"
    """
    Restart the current phase with corrected parameters.
    """

    REPLAN = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Replan"
    )
    """
    Select a new phase sequence from the updated state.
    """

    ABORT = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Abort"
    )
    """
    Terminate the manipulation safely.
    """


class RegulationMode(StrEnum):
    """
    How a controller regulates its commands within a phase.
    """

    FEEDFORWARD = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Feedforward"
    """
    Commands are selected from predictions before sensory correction.
    """

    CONTINUOUS_FEEDBACK = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContinuousFeedback"
    """
    Commands are continuously corrected from monitored sensory signals.
    """

    HYBRID = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Hybrid"
    )
    """
    Feedforward predictions and continuous sensory correction cooperate.
    """


class SensoryModality(StrEnum):
    """
    The sensory channel carrying evidence of an event.
    """

    TACTILE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Tactile"
    )
    """
    Signals from the contact surfaces of the effector, the primary channel for contact events.
    """

    PROPRIOCEPTIVE = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Proprioceptive"
    """
    Signals of limb configuration and effort.
    """

    VISUAL = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Visual"
    )
    """
    Exteroceptive visual signals, also used to anticipate contact before it occurs.
    """

    AUDITORY = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Auditory"
    )
    """
    Acoustic evidence of an event or of a continuously changing state.
    """


class TrajectoryTrend(StrEnum):
    """
    The direction of change expected across a signal trajectory.
    """

    INCREASING = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Increasing"
    """
    The signal magnitude rises over the prediction interval.
    """

    DECREASING = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Decreasing"
    """
    The signal magnitude falls over the prediction interval.
    """

    STABLE = (
        "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Stable"
    )
    """
    The signal remains within a bounded steady range.
    """


class UpdateTimescale(StrEnum):
    """
    How many interactions an update of represented object properties needs.
    """

    SINGLE_TRIAL = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SingleTrial"
    """
    One interaction suffices, as for natural objects with common loads.
    """

    REPEATED_TRIALS = "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RepeatedTrials"
    """
    Several interactions are needed, as with misleading cues, unfamiliar objects or unusual dynamics.
    """


# %% classes


@dataclass(eq=False)
class ActionPhase(DescriptionEntity):
    """
    A building block of a manipulation task: it pursues one subgoal, ends at boundary events and is executed by one controller.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ManipulationPhase",
            "http://aicor.knowledge/l2-ontology.owl#Phase",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Task",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#aimsFor",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Subgoal"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#controlledBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhaseController"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CouplingRegime"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#involvesRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    aims_for: Optional[Subgoal] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#aimsFor",
            functional=True,
        ).as_dict(),
    )
    """
    The single subgoal that individuates the phase.
    """

    controlled_by: Optional[ActionPhaseController] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#controlledBy",
            functional=True,
        ).as_dict(),
    )
    """
    The action-phase controller that executes the phase.
    """

    has_active_parameter: List[Parameter] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasActiveParameter",
            functional=False,
        ).as_dict(),
    )
    """
    A parameter in force during the phase, which its controller is parameterised by in advance.
    """

    has_condition_requirement: List[ConditionRequirement] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionRequirement",
            functional=False,
        ).as_dict(),
    )
    """
    What the phase demands of the world at entry, throughout or at exit.
    """

    has_coupling_regime: Optional[CouplingRegime] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCouplingRegime",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the effector and the manipulandum move as one unit during the phase.
    """

    initiated_by: List[PhaseBoundaryEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#initiatedBy",
            functional=False,
        ).as_dict(),
    )
    """
    An event that begins the phase. The first phase of a structure has none.
    """

    involves_role: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#involvesRole",
            functional=False,
        ).as_dict(),
    )
    """
    A participant role taking part in the phase.
    """

    is_optional: Optional[bool] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#isOptional",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the phase is skipped when the initial or desired state already satisfies its subgoal.
    """

    realizes_action_category: List[ConceptReference] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#realizesActionCategory",
            functional=False,
        ).as_dict(),
    )
    """
    A CRAM-AICOR action category the phase carries out, as its IRI, such as aicor:Tilting.
    """

    terminated_by: List[PhaseBoundaryEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatedBy",
            functional=False,
        ).as_dict(),
    )
    """
    An event that ends the phase. A phase without a boundary is not a phase.
    """


@dataclass(eq=False)
class ActionPhaseController(DescriptionEntity):
    """
    A learned control policy that executes one phase: it is parameterised in advance, generates motor commands, predicts the sensory events marking its subgoal and holds corrective programmes.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhaseController",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#RobotControlLaw",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Method",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-01",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-04",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasRegulationMode",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RegulationMode"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsSensoryEvent",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    governed_by: List[InternalModel] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#governedBy",
            functional=False,
        ).as_dict(),
    )
    """
    An internal model the controller uses to predict and to generate commands.
    """

    has_continuous_prediction: List[ContinuousSignalPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasContinuousPrediction",
            functional=False,
        ).as_dict(),
    )
    """
    A continuous sensory prediction the controller compares throughout its phase.
    """

    has_corrective_program: List[CorrectiveAction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCorrectiveProgram",
            functional=False,
        ).as_dict(),
    )
    """
    A corrective programme the controller runs when a matching mismatch is detected.
    """

    has_regulation_mode: Optional[RegulationMode] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasRegulationMode",
            functional=True,
        ).as_dict(),
    )
    """
    How the controller regulates its commands within the phase, independently of how the phase ends.
    """

    monitors_modality: List[SensoryModality] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#monitorsModality",
            functional=False,
        ).as_dict(),
    )
    """
    A sensory channel the controller tracks throughout its phase rather than sampling at a boundary.
    """

    predicts_sensory_event: List[SensoryEventPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsSensoryEvent",
            functional=False,
        ).as_dict(),
    )
    """
    A sensory event the controller predicts, in particular those that signify attainment of its subgoal.
    """


@dataclass(eq=False)
class ActionPhaseStructure(DescriptionEntity):
    """
    One hypothesis of how a kind of manipulation task is organised: its roles, phases, controllers, subgoals, boundary events, predictions and comparisons. The task's sensory plan and the unit a language model generates.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhaseStructure",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ForceDynamicSchema",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Workflow",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-001",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-02",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#decomposesTask",
                quantifier=Quantifier.EXACTLY,
                cardinality=1,
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesEventType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesSubgoal",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Subgoal"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesTransition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseTransition"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    decomposes_task: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#decomposesTask",
            functional=True,
        ).as_dict(),
    )
    """
    The task divided into phases, as the IRI of a CRAM-AICOR action category, such as aicor:Pouring.
    """

    defines_comparison: List[PredictionComparison] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesComparison",
            functional=False,
        ).as_dict(),
    )
    """
    A comparison between a prediction and an observation.
    """

    defines_component: List[DescriptionEntity] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesComponent",
            functional=False,
        ).as_dict(),
    )
    """
    A fragment the structure declares; a structure refers only to fragments it declares.
    """

    defines_condition_type: List[PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesConditionType",
            functional=False,
        ).as_dict(),
    )
    """
    A world-state type the structure's phases demand, bring about or protect.
    """

    defines_continuous_prediction: List[ContinuousSignalPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesContinuousPrediction",
            functional=False,
        ).as_dict(),
    )
    """
    A continuous prediction compared throughout a phase.
    """

    defines_control_point: List[SensorimotorControlPoint] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesControlPoint",
            functional=False,
        ).as_dict(),
    )
    """
    A sensorimotor checkpoint at a mechanical event.
    """

    defines_control_point_exemption: List[BoundaryControlExemption] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesControlPointExemption",
            functional=False,
        ).as_dict(),
    )
    """
    A boundary that is justified in having no sensorimotor checkpoint.
    """

    defines_controller: List[ActionPhaseController] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesController",
            functional=False,
        ).as_dict(),
    )
    """
    An action-phase controller that executes one of the structure's phases.
    """

    defines_correction: List[CorrectiveAction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesCorrection",
            functional=False,
        ).as_dict(),
    )
    """
    A corrective action programme of one of the controllers.
    """

    defines_event_type: List[PhaseBoundaryEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesEventType",
            functional=False,
        ).as_dict(),
    )
    """
    A boundary event type that begins, ends or gates the structure's phases.
    """

    defines_force_opposition: List[ForceOpposition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesForceOpposition",
            functional=False,
        ).as_dict(),
    )
    """
    A force opposition resolved at one of the structure's events.
    """

    defines_initial_state_estimate: List[InitialStateEstimate] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesInitialStateEstimate",
            functional=False,
        ).as_dict(),
    )
    """
    A next-phase state supplied by a successful comparison.
    """

    defines_model_update: List[ModelUpdate] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesModelUpdate",
            functional=False,
        ).as_dict(),
    )
    """
    A revision of represented estimates driven by prediction errors.
    """

    defines_observation_type: List[SensoryObservationType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesObservationType",
            functional=False,
        ).as_dict(),
    )
    """
    A sensory observation type available for comparison.
    """

    defines_parameter: List[Parameter] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesParameter",
            functional=False,
        ).as_dict(),
    )
    """
    An estimate, setpoint, threshold or margin the phases are parameterised by.
    """

    defines_phase: List[ActionPhase] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPhase",
            functional=False,
        ).as_dict(),
    )
    """
    A phase the task is divided into.
    """

    defines_prediction: List[SensoryEventPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPrediction",
            functional=False,
        ).as_dict(),
    )
    """
    A sensory prediction expected at a mechanical event.
    """

    defines_prediction_error: List[PredictionError] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesPredictionError",
            functional=False,
        ).as_dict(),
    )
    """
    A prediction error a mismatch can be classified as.
    """

    defines_role: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesRole",
            functional=False,
        ).as_dict(),
    )
    """
    A participant slot of the task.
    """

    defines_sensory_feature: List[SensoryFeature] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesSensoryFeature",
            functional=False,
        ).as_dict(),
    )
    """
    A sensory feature the structure's predictions and observations refer to.
    """

    defines_subgoal: List[Subgoal] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesSubgoal",
            functional=False,
        ).as_dict(),
    )
    """
    A subgoal one of the structure's phases pursues.
    """

    defines_transition: List[PhaseTransition] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#definesTransition",
            functional=False,
        ).as_dict(),
    )
    """
    A directed transition between two of the structure's phases.
    """

    defines_gaze_fixation: List[DescriptionEntity] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/gaze#definesGazeFixation",
            functional=False,
        ).as_dict(),
    )
    """
    A gaze fixation that supports one of the structure's phases.
    """


@dataclass(eq=False)
class ManipulationRole(DescriptionEntity):
    """
    A participant slot of a phase structure. At execution an aicor:RoleBinding binds a world object to it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ParticipantRole",
            "http://aicor.knowledge/l2-ontology.owl#RoleSlot",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Role",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#minimumFillerCount",
                quantifier=Quantifier.MAXIMUM,
                cardinality=1,
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    minimum_filler_count: Optional[int] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#minimumFillerCount",
            functional=True,
        ).as_dict(),
    )
    """
    How many world entities must fill the slot; one when not stated.
    """


@dataclass(eq=False)
class Actor(ManipulationRole):
    """
    The agent performing the manipulation. Distinct from the effector it controls.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Actor",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#AgentRole",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Parameter(DescriptionEntity):
    """
    A quantitative or categorical slot in a phase structure, applying to participant roles.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Parameter",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Parameter",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Parameter",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasValueForm",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ParameterValueForm"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#parameterOnRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_categorical_value: Optional[CategoricalValue] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCategoricalValue",
            functional=True,
        ).as_dict(),
    )
    """
    The value of a categorical parameter.
    """

    has_parameter_range: Optional[MagnitudeRange] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
            functional=True,
        ).as_dict(),
    )
    """
    The value of a quantitative parameter.
    """

    has_physical_dimension: Optional[PhysicalDimension] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
            functional=True,
        ).as_dict(),
    )
    """
    The physical dimension of a quantitative parameter, which the unit of its range must measure.
    """

    has_value_form: Optional[ParameterValueForm] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasValueForm",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the parameter holds a magnitude range or a categorical value.
    """

    parameter_on_role: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#parameterOnRole",
            functional=False,
        ).as_dict(),
    )
    """
    A participant role to which the parameter applies.
    """


@dataclass(eq=False)
class QuantitativeParameter(Parameter):
    """
    A parameter holding a magnitude range in a physical dimension, and never a categorical value.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            ComplementOf(
                operand=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCategoricalValue",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
                )
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MagnitudeRange"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhysicalDimension"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Parameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasValueForm",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Quantitative",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class AngleParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are angles, measured in a unit of angle.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AngleParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Angle",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Angle",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class AngularRateParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are angular rates, measured in a unit of angular rate.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AngularRateParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AngularRate",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AngularRate",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class BoundaryControlExemption(DescriptionEntity):
    """
    An explicit reason why a phase boundary has no sensorimotor control point.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#BoundaryControlExemption",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#exemptsEvent",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasJustification",
                quantifier=Quantifier.EXACTLY,
                cardinality=1,
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    exempts_event: Optional[PhaseBoundaryEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#exemptsEvent",
            functional=True,
        ).as_dict(),
    )
    """
    The phase boundary intentionally represented without a control point.
    """

    has_justification: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasJustification",
            functional=True,
        ).as_dict(),
    )
    """
    The policy, instruction or scope decision justifying the exemption, given as the IRI of a concept.
    """


@dataclass(eq=False)
class CategoricalParameter(Parameter):
    """
    A parameter holding a categorical value, and never a fabricated numerical range.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CategoricalParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            ComplementOf(
                operand=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
                )
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCategoricalValue",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CategoricalValue"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Parameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasValueForm",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Categorical",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class CategoricalValue(DescriptionEntity):
    """
    A value that names a category instead of giving a number.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CategoricalValue",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Region",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Region",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCategory",
                quantifier=Quantifier.EXACTLY,
                cardinality=1,
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_category: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCategory",
            functional=True,
        ).as_dict(),
    )
    """
    The term identifying the category, given as the IRI of a concept.
    """


@dataclass(eq=False)
class ConditionRequirement(DescriptionEntity):
    """
    A phase's demand on a condition: when it applies, whether it must hold or be absent, and whether it is presupposed.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionRequirement",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Relation",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Relation",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#hasConditionType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionMode",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionMode"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionPolarity",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionPolarity"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_condition_mode: Optional[ConditionMode] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionMode",
            functional=True,
        ).as_dict(),
    )
    """
    When, relative to the phase, the condition is demanded.
    """

    has_condition_polarity: Optional[ConditionPolarity] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionPolarity",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the condition is demanded to hold or to be absent.
    """

    is_world_assumption: Optional[bool] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#isWorldAssumption",
            functional=True,
        ).as_dict(),
    )
    """
    Whether no phase establishes the condition, so that it is presupposed of the world.
    """

    violation_signalled_by: Optional[FailureEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#violationSignalledBy",
            functional=True,
        ).as_dict(),
    )
    """
    The failure event that registers the condition ceasing to hold.
    """

    has_condition_type: List[PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#hasConditionType",
            functional=False,
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#hasConditionType.
    """


@dataclass(eq=False)
class PhaseConditionType(DescriptionEntity):
    """
    A type of world state relevant to a phase, about one or more participant roles.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#FluentType",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#EventType",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#conditionOnRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    condition_on_role: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#conditionOnRole",
            functional=False,
        ).as_dict(),
    )
    """
    A role the condition is about. Without it the condition cannot be bound to the world.
    """


@dataclass(eq=False)
class ContactCondition(PhaseConditionType):
    """
    The condition that two participants are mechanically in contact. Contact events establish or break it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContactCondition",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PhaseBoundaryEventType(DescriptionEntity):
    """
    A type of discrete transition that can begin, end or gate a phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#EventType",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#EventType",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    terminates_state_type: List[PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
            functional=False,
        ).as_dict(),
    )
    """
    A condition that stops holding when an event of this type occurs. The counterpart of aicor:initiatesStateType.
    """

    initiates_state_type: List[PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
            functional=False,
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#initiatesStateType.
    """


@dataclass(eq=False)
class MechanicalEventType(PhaseBoundaryEventType):
    """
    A type of physical transition between participants that the controller can predict and detect. Contact change and threshold crossing may coincide.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#ForceDynamicEventType",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#classifies",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://aicor.knowledge/l2-ontology.owl#ForceDynamicEvent"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    ends_force: List[ForceComponent] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#endsForce",
            functional=False,
        ).as_dict(),
    )
    """
    A force that stops acting at the event, such as the grip force at release.
    """

    resolves_opposition: Optional[ForceOpposition] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#resolvesOpposition",
            functional=True,
        ).as_dict(),
    )
    """
    The force opposition whose driving force overcomes its resisting force at the event.
    """

    starts_force: List[ForceComponent] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#startsForce",
            functional=False,
        ).as_dict(),
    )
    """
    A force that begins to act at the event, such as the support reaction at touchdown.
    """

    between_role_fillers: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
            functional=False,
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers.
    """


@dataclass(eq=False)
class ContactEventType(MechanicalEventType):
    """
    A mechanical event that makes or breaks contact.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContactEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
        ),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType"
                ),
                UnionOf(
                    operands=(
                        PropertyRestriction(
                            property_iri="http://aicor.knowledge/l2-ontology.owl#initiatesStateType",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContactCondition"
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#terminatesStateType",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContactCondition"
                            ),
                        ),
                    )
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Estimate(Parameter):
    """
    A belief used to predict or to scale manipulation control.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    predicted_from: List[PredictionSource] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictedFrom",
            functional=False,
        ).as_dict(),
    )
    """
    A source from which the estimate is predicted before or at contact.
    """

    scales_setpoint: List[MotorSetpoint] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#scalesSetpoint",
            functional=False,
        ).as_dict(),
    )
    """
    A motor setpoint the estimate scales feedforward, before any sensory confirmation is available.
    """


@dataclass(eq=False)
class ContactPropertyEstimate(Estimate):
    """
    A belief about a property of an effector-object contact interface, such as its friction.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContactPropertyEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ContinuousSignalPrediction(DescriptionEntity):
    """
    A predicted sensory trajectory compared with observation throughout a phase to adjust a setpoint.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContinuousSignalPrediction",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Expectation",
            "http://aicor.knowledge/l2-ontology.owl#Prediction",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-008",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#generatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForwardModel"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#comparedWith",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryObservationType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#controlsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Setpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryModality"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasTolerance",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MagnitudeRange"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsTrajectory",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SignalTrajectory"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    compared_with: Optional[SensoryObservationType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#comparedWith",
            functional=True,
        ).as_dict(),
    )
    """
    The observation type continuously compared with the prediction.
    """

    controls_setpoint: Optional[Setpoint] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#controlsSetpoint",
            functional=True,
        ).as_dict(),
    )
    """
    The setpoint adjusted from the comparison error.
    """

    has_modality: Optional[SensoryModality] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
            functional=True,
        ).as_dict(),
    )
    """
    The sensory channel in which the prediction or observation is carried.
    """

    has_tolerance: Optional[MagnitudeRange] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasTolerance",
            functional=True,
        ).as_dict(),
    )
    """
    The admissible deviation from the predicted magnitude or trajectory.
    """

    predicts_trajectory: Optional[SignalTrajectory] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsTrajectory",
            functional=True,
        ).as_dict(),
    )
    """
    The structured feature change expected during the phase.
    """

    generated_by: List[InternalModel] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#generatedBy",
            functional=False,
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#generatedBy.
    """


@dataclass(eq=False)
class CorrectiveAction(DescriptionEntity):
    """
    A corrective programme specific to its controller and the kind of mismatch, protecting or completing the current action.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CorrectiveAction",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Task",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-005",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-03",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-04",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCorrectionDirection",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CorrectionDirection"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasRecoveryMode",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#RecoveryMode"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#intendedEffect",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredByError",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionError"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    correction_terminated_by: List[PhaseBoundaryEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#correctionTerminatedBy",
            functional=False,
        ).as_dict(),
    )
    """
    An event that ends a corrective programme, such as lift-off ending probing force increases.
    """

    has_correction_direction: Optional[CorrectionDirection] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasCorrectionDirection",
            functional=True,
        ).as_dict(),
    )
    """
    How the targeted setpoints are changed.
    """

    has_recovery_mode: Optional[RecoveryMode] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasRecoveryMode",
            functional=True,
        ).as_dict(),
    )
    """
    How execution proceeds when the response is released.
    """

    has_response_latency: Optional[TemporalWindow] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResponseLatency",
            functional=True,
        ).as_dict(),
    )
    """
    The delay before the response changes the motor output: about 100 ms for tactile, over 200 ms for visual triggers in humans.
    """

    intended_effect: List[PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#intendedEffect",
            functional=False,
        ).as_dict(),
    )
    """
    A condition the correction is meant to establish or preserve.
    """

    targets_setpoint: List[MotorSetpoint] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsSetpoint",
            functional=False,
        ).as_dict(),
    )
    """
    A motor setpoint the response changes.
    """

    triggered_by_error: List[PredictionError] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredByError",
            functional=False,
        ).as_dict(),
    )
    """
    A prediction error that releases the response or supplies evidence for the update.
    """


@dataclass(eq=False)
class DimensionlessParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are ratios or fractions without a physical unit.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DimensionlessParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Dimensionless",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Dimensionless",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class DynamicViscosityParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are dynamic viscosities, measured in a unit of dynamic viscosity.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DynamicViscosityParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DynamicViscosity",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DynamicViscosity",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class DynamicsEstimate(Estimate):
    """
    A belief about how an object state evolves under action.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#DynamicsEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class EntryRequirement(ConditionRequirement):
    """
    A condition demanded for the phase to begin: the classical precondition.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EntryRequirement",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionRequirement"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionMode",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AtEntry",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InternalModel(DescriptionEntity):
    """
    The controller's own representation of body and environment, used to predict and to invert.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InternalModel",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class EstimateRepresentation(InternalModel):
    """
    Represented content that holds one or more estimates. Not presumed disjoint from the models that use it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EstimateRepresentation",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    holds_estimate: List[Estimate] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#holdsEstimate",
            functional=False,
        ).as_dict(),
    )
    """
    An estimate stored as content of the representation.
    """


@dataclass(eq=False)
class ExitRequirement(ConditionRequirement):
    """
    A condition brought about by completing the phase: the classical postcondition.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ExitRequirement",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionRequirement"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionMode",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AtExit",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ExternalTerminationEventType(PhaseBoundaryEventType):
    """
    A phase boundary imposed from outside, such as an instruction or a safety condition.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ExternalTerminationEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredBy",
                quantifier=Quantifier.EXACTLY,
                cardinality=1,
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    triggered_by: Optional[ConceptReference] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredBy",
            functional=True,
        ).as_dict(),
    )
    """
    The external request, policy or safety condition that ends the phase.
    """


@dataclass(eq=False)
class FailureEventType(PhaseBoundaryEventType):
    """
    A boundary event that marks failure rather than intended completion.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FailureEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class FeedbackRegulatedController(ActionPhaseController):
    """
    A controller that corrects its commands from sensory signals while its phase runs; it monitors at least one modality.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FeedbackRegulatedController",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#monitorsModality",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryModality"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhaseController"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasRegulationMode",
                    quantifier=Quantifier.SOME,
                    filler=EnumeratedClass(
                        members=(
                            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ContinuousFeedback",
                            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Hybrid",
                        )
                    ),
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ForceComponent(DescriptionEntity):
    """
    A force of one kind acting on a participant, classifying the physical forces of an execution.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceComponent",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Concept",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#classifies",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(iri="http://aicor.knowledge/l2-ontology.owl#Force"),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#actsOnRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceKind"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    acts_on_role: Optional[ManipulationRole] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#actsOnRole",
            functional=True,
        ).as_dict(),
    )
    """
    The participant the force acts on.
    """

    exerted_by_role: Optional[ManipulationRole] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#exertedByRole",
            functional=True,
        ).as_dict(),
    )
    """
    The participant exerting the force; absent for weight and inertia.
    """

    has_force_kind: Optional[ForceKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasForceKind",
            functional=True,
        ).as_dict(),
    )
    """
    The kind of force, such as weight, grip force or friction.
    """


@dataclass(eq=False)
class ForceOpposition(DescriptionEntity):
    """
    Two forces acting against each other; the event resolving it occurs once the driving force overcomes the resisting one.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceOpposition",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Relation",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Relation",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-06",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-07",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDrivingForce",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceComponent"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResistingForce",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceComponent"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_driving_force: Optional[ForceComponent] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDrivingForce",
            functional=True,
        ).as_dict(),
    )
    """
    The force that prevails once the event has occurred.
    """

    has_resisting_force: Optional[ForceComponent] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasResistingForce",
            functional=True,
        ).as_dict(),
    )
    """
    The force that prevails until the event; the event's threshold is its limit.
    """


@dataclass(eq=False)
class ForceParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are forces, measured in a unit of force.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Force",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Force",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ForceRateParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are rates of force change, measured in a unit of force rate.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceRateParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceRate",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceRate",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Setpoint(Parameter):
    """
    A desired value whose attainment can constitute a control event.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Setpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MotorSetpoint(Setpoint):
    """
    A target the motor controller commands directly, such as a grip force or a tilt rate.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ForceRateSetpoint(ForceRateParameter, MotorSetpoint):
    """
    A commanded rate of force change, scaled in advance to the predicted load.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForceRateSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#rateOf",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    rate_of: Optional[MotorSetpoint] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#rateOf",
            functional=True,
        ).as_dict(),
    )
    """
    The force setpoint whose rate of change is commanded.
    """


@dataclass(eq=False)
class ForwardModel(InternalModel):
    """
    Maps motor commands, through an efference copy, to their expected sensory consequences.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForwardModel",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InitialStateEstimate(DescriptionEntity):
    """
    The state predicted at the end of one phase, supplied as the initial state of the next phase's controller.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InitialStateEstimate",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-007",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#basedOnObservation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryObservationType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#estimatesCondition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#forPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    based_on_observation: List[SensoryObservationType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#basedOnObservation",
            functional=False,
        ).as_dict(),
    )
    """
    An observation supporting the state estimate.
    """

    estimates_condition: List[PhaseConditionType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#estimatesCondition",
            functional=False,
        ).as_dict(),
    )
    """
    A condition estimated to hold at phase entry.
    """

    for_phase: Optional[ActionPhase] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#forPhase",
            functional=True,
        ).as_dict(),
    )
    """
    The phase whose initial state is estimated.
    """


@dataclass(eq=False)
class InvariantRequirement(ConditionRequirement):
    """
    A condition that must hold throughout the phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InvariantRequirement",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#MaintenanceGoal",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Goal",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionRequirement"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionMode",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Throughout",
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionPolarity",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MustHold",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class InverseModel(InternalModel):
    """
    Turns a desired state into the motor command required to reach it.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InverseModel",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class LengthParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are lengths, measured in a unit of length.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#LengthParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Length",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Length",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MagnitudeRange(DescriptionEntity):
    """
    An interval of a quantity with its unit; the lower bound does not exceed the upper bound.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MagnitudeRange",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ParameterInterval",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Amount",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MeasurementUnit"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#lowerBound",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2001/XMLSchema#double"),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#upperBound",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2001/XMLSchema#double"),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_unit: Optional[MeasurementUnit] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
            functional=True,
        ).as_dict(),
    )
    """
    The measurement unit shared by both bounds.
    """

    lower_bound: Optional[float] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#lowerBound",
            functional=True,
        ).as_dict(),
    )
    """
    The inclusive lower bound of the range.
    """

    upper_bound: Optional[float] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#upperBound",
            functional=True,
        ).as_dict(),
    )
    """
    The inclusive upper bound of the range.
    """


@dataclass(eq=False)
class Manipulandum(ManipulationRole):
    """
    The object whose state the manipulation is about.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Manipulandum",
        linked_terms=("http://aicor.knowledge/l2-ontology.owl#PatientRole",),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MassFlowRateParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are mass flow rates, measured in a unit of mass flow rate.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassFlowRateParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassFlowRate",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassFlowRate",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MassParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are masses, measured in a unit of mass.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MassParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Mass",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Mass",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PredictionComparison(DescriptionEntity):
    """
    The comparison of one prediction with its observation at a control point.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionComparison",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Diagnosis",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-005",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#comparesPrediction",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasComparisonOutcome",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ComparisonOutcome"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    compares_observation: Optional[SensoryObservationType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#comparesObservation",
            functional=True,
        ).as_dict(),
    )
    """
    The received observation, absent only when the error is an absent event.
    """

    compares_prediction: Optional[SensoryEventPrediction] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#comparesPrediction",
            functional=True,
        ).as_dict(),
    )
    """
    The sensory prediction being evaluated.
    """

    confirms_subgoal: Optional[Subgoal] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#confirmsSubgoal",
            functional=True,
        ).as_dict(),
    )
    """
    The subgoal a successful comparison confirms.
    """

    has_comparison_outcome: Optional[ComparisonOutcome] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasComparisonOutcome",
            functional=True,
        ).as_dict(),
    )
    """
    Whether the observation matches the prediction.
    """

    has_prediction_error: List[PredictionError] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPredictionError",
            functional=False,
        ).as_dict(),
    )
    """
    An error a mismatching comparison is classified as.
    """

    invokes_correction: List[CorrectiveAction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#invokesCorrection",
            functional=False,
        ).as_dict(),
    )
    """
    A corrective response a mismatch releases.
    """

    produces_model_update: List[ModelUpdate] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#producesModelUpdate",
            functional=False,
        ).as_dict(),
    )
    """
    A representation update a mismatch supports.
    """

    supplies_state_estimate: Optional[InitialStateEstimate] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#suppliesStateEstimate",
            functional=True,
        ).as_dict(),
    )
    """
    The next phase's initial state supplied by a successful comparison.
    """


@dataclass(eq=False)
class MatchingComparison(PredictionComparison):
    """
    A comparison that agrees with the prediction: no error, and it confirms a subgoal or supplies the next initial state.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MatchingComparison",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-005",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-007",
        ),
        restrictions=(
            ComplementOf(
                operand=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPredictionError",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
                )
            ),
            UnionOf(
                operands=(
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#confirmsSubgoal",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Subgoal"
                        ),
                    ),
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#suppliesStateEstimate",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#InitialStateEstimate"
                        ),
                    ),
                )
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionComparison"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasComparisonOutcome",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Match",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MechanicalThreshold(Parameter):
    """
    A boundary value whose crossing constitutes a mechanical event.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalThreshold",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class MismatchingComparison(PredictionComparison):
    """
    A comparison that disagrees with the prediction: it classifies an error, confirms no subgoal, and releases a correction or model update.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MismatchingComparison",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-005",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
        ),
        restrictions=(
            ComplementOf(
                operand=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#confirmsSubgoal",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
                )
            ),
            ComplementOf(
                operand=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#suppliesStateEstimate",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(iri="http://www.w3.org/2002/07/owl#Thing"),
                )
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPredictionError",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionError"
                ),
            ),
            UnionOf(
                operands=(
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#invokesCorrection",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#CorrectiveAction"
                        ),
                    ),
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#producesModelUpdate",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ModelUpdate"
                        ),
                    ),
                )
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionComparison"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasComparisonOutcome",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Mismatch",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ModelUpdate(DescriptionEntity):
    """
    A revision of represented estimates that changes later predictions with the same object.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ModelUpdate",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#EventType",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-09",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUpdateDirection",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EstimateUpdateDirection"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredByError",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionError"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#updatesEstimate",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Estimate"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#updatesRepresentation",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EstimateRepresentation"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_update_direction: Optional[EstimateUpdateDirection] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUpdateDirection",
            functional=True,
        ).as_dict(),
    )
    """
    How the represented estimates change.
    """

    has_update_timescale: Optional[UpdateTimescale] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUpdateTimescale",
            functional=True,
        ).as_dict(),
    )
    """
    Whether one interaction suffices for the update or several are needed.
    """

    triggered_by_error: List[PredictionError] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#triggeredByError",
            functional=False,
        ).as_dict(),
    )
    """
    A prediction error that releases the response or supplies evidence for the update.
    """

    updates_estimate: List[Estimate] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#updatesEstimate",
            functional=False,
        ).as_dict(),
    )
    """
    An estimate the revision changes.
    """

    updates_representation: Optional[EstimateRepresentation] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#updatesRepresentation",
            functional=True,
        ).as_dict(),
    )
    """
    The representation in which the content is changed.
    """


@dataclass(eq=False)
class ObjectPropertyEstimate(Estimate):
    """
    A belief about a comparatively stable property of an object, such as its mass or shape.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ObjectPropertyEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectRepresentation(EstimateRepresentation):
    """
    Stored beliefs about an object's physical properties, accumulated over prior manipulations.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ObjectRepresentation",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-009",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class ObjectStateEstimate(Estimate):
    """
    A belief about a changeable state of an object, such as its fill level.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ObjectStateEstimate",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PhaseTransition(DescriptionEntity):
    """
    A gated edge of the phase sequence. Where comparisons permit it, only matching comparisons do.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseTransition",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Relation",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Relation",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#fromPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#gatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#permittedBy",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MatchingComparison"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#toPhase",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ActionPhase"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    from_phase: Optional[ActionPhase] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#fromPhase",
            functional=True,
        ).as_dict(),
    )
    """
    The phase control leaves.
    """

    gated_by: List[PhaseBoundaryEventType] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#gatedBy",
            functional=False,
        ).as_dict(),
    )
    """
    A boundary event that releases the transition.
    """

    permitted_by: List[PredictionComparison] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#permittedBy",
            functional=False,
        ).as_dict(),
    )
    """
    A successful comparison required in addition to the event gates.
    """

    to_phase: Optional[ActionPhase] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#toPhase",
            functional=True,
        ).as_dict(),
    )
    """
    The phase control enters.
    """


@dataclass(eq=False)
class SetpointAttainmentEventType(PhaseBoundaryEventType):
    """
    A commanded pose, force or task quantity has been reached. Not necessarily a mechanical transition.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SetpointAttainmentEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Setpoint"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    attains_setpoint: List[Setpoint] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainsSetpoint",
            functional=False,
        ).as_dict(),
    )
    """
    A setpoint whose attainment constitutes the event.
    """

    between_role_fillers: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers",
            functional=False,
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers.
    """


@dataclass(eq=False)
class PoseAttainmentEventType(SetpointAttainmentEventType):
    """
    A commanded pose has been reached: a change of position, not of contact or force.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PoseAttainmentEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class PredictionError(DescriptionEntity):
    """
    A classified discrepancy between a prediction and an observation.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionError",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Diagnosis",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-005",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-010",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasErrorKind",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionErrorKind"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_discrepancy: Optional[MagnitudeRange] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasDiscrepancy",
            functional=True,
        ).as_dict(),
    )
    """
    The size of the discrepancy, when the error is quantitative.
    """

    has_error_kind: Optional[PredictionErrorKind] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasErrorKind",
            functional=True,
        ).as_dict(),
    )
    """
    The dimension along which the prediction failed.
    """


@dataclass(eq=False)
class ProhibitionRequirement(ConditionRequirement):
    """
    A condition that must be absent throughout the phase.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ProhibitionRequirement",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#PreventionGoal",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Goal",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionRequirement"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionMode",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Throughout",
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasConditionPolarity",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MustNotHold",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SafetyMargin(QuantitativeParameter):
    """
    The excess of a motor setpoint over a threshold it must stay clear of, typically 10-40 percent for grip force.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SafetyMargin",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-025",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-06",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#marginOn",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MotorSetpoint"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#marginOver",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalThreshold"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    margin_on: Optional[MotorSetpoint] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#marginOn",
            functional=True,
        ).as_dict(),
    )
    """
    The motor setpoint carrying the safety margin.
    """

    margin_over: Optional[MechanicalThreshold] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#marginOver",
            functional=True,
        ).as_dict(),
    )
    """
    The threshold below the setpoint.
    """


@dataclass(eq=False)
class SensorimotorControlPoint(DescriptionEntity):
    """
    A planned event at which predicted and actual sensory signals are compared to monitor progress and detect errors.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensorimotorControlPoint",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ProcessMonitor",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-001",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-005",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-01",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#atEventType",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasComparison",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PredictionComparison"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPrediction",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    at_event_type: Optional[MechanicalEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#atEventType",
            functional=True,
        ).as_dict(),
    )
    """
    The mechanical event at which comparisons are performed.
    """

    has_comparison: List[PredictionComparison] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasComparison",
            functional=False,
        ).as_dict(),
    )
    """
    A possible match or mismatch outcome at the control point.
    """

    has_prediction: List[SensoryEventPrediction] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPrediction",
            functional=False,
        ).as_dict(),
    )
    """
    A sensory consequence expected at the event, which the control point checks.
    """


@dataclass(eq=False)
class SensoryEventPrediction(DescriptionEntity):
    """
    A sensory event a forward model expects at a phase boundary event, predicting an event or a feature but not both.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Expectation",
            "http://aicor.knowledge/l2-ontology.owl#Prediction",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-002",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-02",
        ),
        restrictions=(
            ComplementOf(
                operand=IntersectionOf(
                    operands=(
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsEvent",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.w3.org/2002/07/owl#Thing"
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsFeature",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.w3.org/2002/07/owl#Thing"
                            ),
                        ),
                    )
                )
            ),
            PropertyRestriction(
                property_iri="http://aicor.knowledge/l2-ontology.owl#generatedBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ForwardModel"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#causedByEvent",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#expectedOnRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasExpectedTiming",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TemporalWindow"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryModality"
                ),
            ),
            UnionOf(
                operands=(
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsEvent",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                        ),
                    ),
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsFeature",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryFeature"
                        ),
                    ),
                )
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    caused_by_event: Optional[PhaseBoundaryEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#causedByEvent",
            functional=True,
        ).as_dict(),
    )
    """
    The event expected to produce the sensory consequence.
    """

    expected_on_role: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#expectedOnRole",
            functional=False,
        ).as_dict(),
    )
    """
    A participant role at which the consequence is expected.
    """

    has_expected_magnitude: Optional[MagnitudeRange] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasExpectedMagnitude",
            functional=True,
        ).as_dict(),
    )
    """
    The expected magnitude, when the prediction is quantitative.
    """

    has_expected_timing: Optional[TemporalWindow] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasExpectedTiming",
            functional=True,
        ).as_dict(),
    )
    """
    The interval in which the consequence or trajectory is expected.
    """

    has_modality: Optional[SensoryModality] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
            functional=True,
        ).as_dict(),
    )
    """
    The sensory channel in which the prediction or observation is carried.
    """

    predicts_event: Optional[PhaseBoundaryEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsEvent",
            functional=True,
        ).as_dict(),
    )
    """
    The expected event.
    """

    predicts_feature: Optional[SensoryFeature] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#predictsFeature",
            functional=True,
        ).as_dict(),
    )
    """
    The expected signal feature.
    """

    signals_subgoal: Optional[Subgoal] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#signalsSubgoal",
            functional=True,
        ).as_dict(),
    )
    """
    The subgoal whose attainment the predicted sensory event signifies.
    """

    generated_by: List[InternalModel] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://aicor.knowledge/l2-ontology.owl#generatedBy",
            functional=False,
        ).as_dict(),
    )
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#generatedBy.
    """


@dataclass(eq=False)
class SensoryFeature(DescriptionEntity):
    """
    A detectable property of a sensory signal, such as contact timing, force direction, friction or a sound.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryFeature",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Concept",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Concept",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
        ),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SensoryGoal(SensoryEventPrediction):
    """
    A predicted sensory event that signifies a subgoal has been attained.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryGoal",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#JF09-02",
        ),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#signalsSubgoal",
                    quantifier=Quantifier.SOME,
                    filler=NamedClass(
                        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Subgoal"
                    ),
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SensoryObservationType(DescriptionEntity):
    """
    A sensory result compared with a prediction, observing an event or a feature but not both.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryObservationType",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#ObservationEvent",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#EventType",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EmpiricalFinding",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
        ),
        restrictions=(
            ComplementOf(
                operand=IntersectionOf(
                    operands=(
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesEvent",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.w3.org/2002/07/owl#Thing"
                            ),
                        ),
                        PropertyRestriction(
                            property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesFeature",
                            quantifier=Quantifier.SOME,
                            filler=NamedClass(
                                iri="http://www.w3.org/2002/07/owl#Thing"
                            ),
                        ),
                    )
                )
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryModality"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observedAt",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TemporalWindow"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observedOnRole",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ManipulationRole"
                ),
            ),
            UnionOf(
                operands=(
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesEvent",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                        ),
                    ),
                    PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesFeature",
                        quantifier=Quantifier.SOME,
                        filler=NamedClass(
                            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryFeature"
                        ),
                    ),
                )
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_modality: Optional[SensoryModality] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
            functional=True,
        ).as_dict(),
    )
    """
    The sensory channel in which the prediction or observation is carried.
    """

    has_observed_magnitude: Optional[MagnitudeRange] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasObservedMagnitude",
            functional=True,
        ).as_dict(),
    )
    """
    The observed magnitude, when the observation is quantitative.
    """

    observed_at: Optional[TemporalWindow] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observedAt",
            functional=True,
        ).as_dict(),
    )
    """
    The interval in which the observation occurs.
    """

    observed_on_role: List[ManipulationRole] = field(
        default_factory=list,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observedOnRole",
            functional=False,
        ).as_dict(),
    )
    """
    A participant role involved in the observation.
    """

    observes_event: Optional[PhaseBoundaryEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesEvent",
            functional=True,
        ).as_dict(),
    )
    """
    The observed event.
    """

    observes_feature: Optional[SensoryFeature] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesFeature",
            functional=True,
        ).as_dict(),
    )
    """
    The observed feature.
    """


@dataclass(eq=False)
class SignalTrajectory(DescriptionEntity):
    """
    A structured change in one sensory feature over a time interval.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SignalTrajectory",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Description",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasExpectedRange",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MagnitudeRange"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasSignalFeature",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryFeature"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasTrend",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TrajectoryTrend"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    has_expected_range: Optional[MagnitudeRange] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasExpectedRange",
            functional=True,
        ).as_dict(),
    )
    """
    The range the feature traverses or stays within.
    """

    has_signal_feature: Optional[SensoryFeature] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasSignalFeature",
            functional=True,
        ).as_dict(),
    )
    """
    The sensory feature whose evolution is described.
    """

    has_trend: Optional[TrajectoryTrend] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasTrend",
            functional=True,
        ).as_dict(),
    )
    """
    The expected direction of the feature's change.
    """


@dataclass(eq=False)
class SourceClaim(DescriptionEntity):
    """
    A statement made by a source document, identified so that concepts can cite it and carry its evidence status.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SourceClaim",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_InformationObject",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#InformationObject",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class SourceDocument(DescriptionEntity):
    """
    A publication whose claims the microworld represents.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SourceDocument",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_InformationObject",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#InformationObject",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class Subgoal(DescriptionEntity):
    """
    The world condition a phase exists to bring about. Its attainment event is evidence, not the condition itself.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Subgoal",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#Goal",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Goal",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#AuthorHypothesis",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-003",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainmentSignalledBy",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseBoundaryEventType"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsCondition",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#PhaseConditionType"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    attainment_signalled_by: Optional[PhaseBoundaryEventType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#attainmentSignalledBy",
            functional=True,
        ).as_dict(),
    )
    """
    The event whose registration is evidence that the subgoal is reached.
    """

    targets_condition: Optional[PhaseConditionType] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#targetsCondition",
            functional=True,
        ).as_dict(),
    )
    """
    The condition the phase is meant to bring about.
    """


@dataclass(eq=False)
class TactileObservation(SensoryObservationType):
    """
    A sensed touch signal, which can only register a mechanical event.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TactileObservation",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#observesEvent",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryObservationType"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Tactile",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TactilePrediction(SensoryEventPrediction):
    """
    A predicted touch signal, which only a mechanical event can cause.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TactilePrediction",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SupportedInterpretation",
        claims=(
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-004",
            "http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#FJ-012",
        ),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#causedByEvent",
                quantifier=Quantifier.ONLY,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalEventType"
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#SensoryEventPrediction"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasModality",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Tactile",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TaskSetpoint(Setpoint):
    """
    A desired task outcome, such as a transferred volume, rather than a motor command.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TaskSetpoint",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class TemporalWindow(DescriptionEntity):
    """
    A time interval relative to a reference event, in a time unit.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TemporalWindow",
        linked_terms=(
            "http://aicor.knowledge/l2-ontology.owl#dul_Region",
            "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#TimeInterval",
        ),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#earliestOffset",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2001/XMLSchema#double"),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Time",
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MeasurementUnit"
                ),
            ),
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#latestOffset",
                quantifier=Quantifier.SOME,
                filler=NamedClass(iri="http://www.w3.org/2001/XMLSchema#double"),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    earliest_offset: Optional[float] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#earliestOffset",
            functional=True,
        ).as_dict(),
    )
    """
    The inclusive beginning of the window, relative to its reference event.
    """

    has_unit: Optional[MeasurementUnit] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
            functional=True,
        ).as_dict(),
    )
    """
    The measurement unit shared by both bounds.
    """

    latest_offset: Optional[float] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#latestOffset",
            functional=True,
        ).as_dict(),
    )
    """
    The inclusive end of the window, relative to its reference event.
    """


@dataclass(eq=False)
class ThresholdCrossingEventType(MechanicalEventType):
    """
    A mechanical event that occurs when a mechanical quantity crosses a threshold.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ThresholdCrossingEventType",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
                quantifier=Quantifier.SOME,
                filler=NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#MechanicalThreshold"
                ),
            ),
        ),
        definition=None,
    )
    """
    The ontology class this class stands for.
    """

    crosses_threshold: Optional[MechanicalThreshold] = field(
        default=None,
        metadata=PropertyTerm(
            iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#crossesThreshold",
            functional=True,
        ).as_dict(),
    )
    """
    The threshold crossed at the event.
    """


@dataclass(eq=False)
class TorqueParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are torques, measured in a unit of torque.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#TorqueParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Torque",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Torque",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class VolumeParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are volumes, measured in a unit of volume.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumeParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Volume",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#Volume",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class VolumetricFlowRateParameter(QuantitativeParameter):
    """
    A quantitative parameter whose values are volumetric flow rates, measured in a unit of volumetric flow rate.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumetricFlowRateParameter",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(
            PropertyRestriction(
                property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasParameterRange",
                quantifier=Quantifier.ONLY,
                filler=PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasUnit",
                    quantifier=Quantifier.ONLY,
                    filler=PropertyRestriction(
                        property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#measuresDimension",
                        quantifier=Quantifier.VALUE,
                        value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumetricFlowRate",
                    ),
                ),
            ),
        ),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#QuantitativeParameter"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#hasPhysicalDimension",
                    quantifier=Quantifier.VALUE,
                    value="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#VolumetricFlowRate",
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


@dataclass(eq=False)
class WorldAssumption(ConditionRequirement):
    """
    A demanded condition that no phase establishes, so it is presupposed of the world.
    """

    term: ClassVar[ClassTerm] = ClassTerm(
        iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#WorldAssumption",
        linked_terms=(),
        evidence_status="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#EngineeringExtension",
        claims=(),
        restrictions=(),
        definition=IntersectionOf(
            operands=(
                NamedClass(
                    iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#ConditionRequirement"
                ),
                PropertyRestriction(
                    property_iri="http://www.ease-crc.org/ont/llm_reasoner/predictive-manipulation/core#isWorldAssumption",
                    quantifier=Quantifier.VALUE,
                    value=True,
                ),
            )
        ),
    )
    """
    The ontology class this class stands for.
    """


# %% property descriptors


@dataclass
class ActsOnRole(PropertyDescriptor):
    """
    The participant the force acts on.
    """


@dataclass
class AimsFor(PropertyDescriptor):
    """
    The single subgoal that individuates the phase.
    """


@dataclass
class AtEventType(PropertyDescriptor):
    """
    The mechanical event at which comparisons are performed.
    """


@dataclass
class AttainmentSignalledBy(PropertyDescriptor):
    """
    The event whose registration is evidence that the subgoal is reached.
    """


@dataclass
class AttainsSetpoint(PropertyDescriptor):
    """
    A setpoint whose attainment constitutes the event.
    """


@dataclass
class BasedOnObservation(PropertyDescriptor):
    """
    An observation supporting the state estimate.
    """


@dataclass
class CausedByEvent(PropertyDescriptor):
    """
    The event expected to produce the sensory consequence.
    """


@dataclass
class ComparedWith(PropertyDescriptor):
    """
    The observation type continuously compared with the prediction.
    """


@dataclass
class ComparesObservation(PropertyDescriptor):
    """
    The received observation, absent only when the error is an absent event.
    """


@dataclass
class ComparesPrediction(PropertyDescriptor):
    """
    The sensory prediction being evaluated.
    """


@dataclass
class ConditionOnRole(PropertyDescriptor):
    """
    A role the condition is about. Without it the condition cannot be bound to the world.
    """


@dataclass
class ConfirmsSubgoal(PropertyDescriptor):
    """
    The subgoal a successful comparison confirms.
    """


@dataclass
class ControlledBy(PropertyDescriptor):
    """
    The action-phase controller that executes the phase.
    """


@dataclass
class ControlsSetpoint(PropertyDescriptor):
    """
    The setpoint adjusted from the comparison error.
    """


@dataclass
class CorrectionTerminatedBy(PropertyDescriptor):
    """
    An event that ends a corrective programme, such as lift-off ending probing force increases.
    """


@dataclass
class CrossesThreshold(PropertyDescriptor):
    """
    The threshold crossed at the event.
    """


@dataclass
class DefinesComponent(PropertyDescriptor):
    """
    A fragment the structure declares; a structure refers only to fragments it declares.
    """


@dataclass
class DefinesComparison(DefinesComponent):
    """
    A comparison between a prediction and an observation.
    """


@dataclass
class DefinesConditionType(DefinesComponent):
    """
    A world-state type the structure's phases demand, bring about or protect.
    """


@dataclass
class DefinesContinuousPrediction(DefinesComponent):
    """
    A continuous prediction compared throughout a phase.
    """


@dataclass
class DefinesControlPoint(DefinesComponent):
    """
    A sensorimotor checkpoint at a mechanical event.
    """


@dataclass
class DefinesControlPointExemption(DefinesComponent):
    """
    A boundary that is justified in having no sensorimotor checkpoint.
    """


@dataclass
class DefinesController(DefinesComponent):
    """
    An action-phase controller that executes one of the structure's phases.
    """


@dataclass
class DefinesCorrection(DefinesComponent):
    """
    A corrective action programme of one of the controllers.
    """


@dataclass
class DefinesEventType(DefinesComponent):
    """
    A boundary event type that begins, ends or gates the structure's phases.
    """


@dataclass
class DefinesForceOpposition(DefinesComponent):
    """
    A force opposition resolved at one of the structure's events.
    """


@dataclass
class DefinesInitialStateEstimate(DefinesComponent):
    """
    A next-phase state supplied by a successful comparison.
    """


@dataclass
class DefinesInternalModel(DefinesComponent):
    """
    An internal model or represented content that governs the controllers.
    """


@dataclass
class DefinesModelUpdate(DefinesComponent):
    """
    A revision of represented estimates driven by prediction errors.
    """


@dataclass
class DefinesObservationType(DefinesComponent):
    """
    A sensory observation type available for comparison.
    """


@dataclass
class DefinesParameter(DefinesComponent):
    """
    An estimate, setpoint, threshold or margin the phases are parameterised by.
    """


@dataclass
class DefinesPhase(DefinesComponent):
    """
    A phase the task is divided into.
    """


@dataclass
class DefinesPrediction(DefinesComponent):
    """
    A sensory prediction expected at a mechanical event.
    """


@dataclass
class DefinesPredictionError(DefinesComponent):
    """
    A prediction error a mismatch can be classified as.
    """


@dataclass
class DefinesRole(DefinesComponent):
    """
    A participant slot of the task.
    """


@dataclass
class DefinesSensoryFeature(DefinesComponent):
    """
    A sensory feature the structure's predictions and observations refer to.
    """


@dataclass
class DefinesSubgoal(DefinesComponent):
    """
    A subgoal one of the structure's phases pursues.
    """


@dataclass
class DefinesTransition(DefinesComponent):
    """
    A directed transition between two of the structure's phases.
    """


@dataclass
class EndsForce(PropertyDescriptor):
    """
    A force that stops acting at the event, such as the grip force at release.
    """


@dataclass
class EstimatesCondition(PropertyDescriptor):
    """
    A condition estimated to hold at phase entry.
    """


@dataclass
class ExemptsEvent(PropertyDescriptor):
    """
    The phase boundary intentionally represented without a control point.
    """


@dataclass
class ExertedByRole(PropertyDescriptor):
    """
    The participant exerting the force; absent for weight and inertia.
    """


@dataclass
class ExpectedOnRole(PropertyDescriptor):
    """
    A participant role at which the consequence is expected.
    """


@dataclass
class ForPhase(PropertyDescriptor):
    """
    The phase whose initial state is estimated.
    """


@dataclass
class FromPhase(PropertyDescriptor):
    """
    The phase control leaves.
    """


@dataclass
class GatedBy(PropertyDescriptor):
    """
    A boundary event that releases the transition.
    """


@dataclass
class GovernedBy(PropertyDescriptor):
    """
    An internal model the controller uses to predict and to generate commands.
    """


@dataclass
class HasActiveParameter(PropertyDescriptor):
    """
    A parameter in force during the phase, which its controller is parameterised by in advance.
    """


@dataclass
class HasCategoricalValue(PropertyDescriptor):
    """
    The value of a categorical parameter.
    """


@dataclass
class HasComparison(PropertyDescriptor):
    """
    A possible match or mismatch outcome at the control point.
    """


@dataclass
class HasConditionRequirement(PropertyDescriptor):
    """
    What the phase demands of the world at entry, throughout or at exit.
    """


@dataclass
class HasContinuousPrediction(PropertyDescriptor):
    """
    A continuous sensory prediction the controller compares throughout its phase.
    """


@dataclass
class HasCorrectiveProgram(PropertyDescriptor):
    """
    A corrective programme the controller runs when a matching mismatch is detected.
    """


@dataclass
class HasDiscrepancy(PropertyDescriptor):
    """
    The size of the discrepancy, when the error is quantitative.
    """


@dataclass
class HasDrivingForce(PropertyDescriptor):
    """
    The force that prevails once the event has occurred.
    """


@dataclass
class HasExpectedMagnitude(PropertyDescriptor):
    """
    The expected magnitude, when the prediction is quantitative.
    """


@dataclass
class HasExpectedRange(PropertyDescriptor):
    """
    The range the feature traverses or stays within.
    """


@dataclass
class HasExpectedTiming(PropertyDescriptor):
    """
    The interval in which the consequence or trajectory is expected.
    """


@dataclass
class HasObservedMagnitude(PropertyDescriptor):
    """
    The observed magnitude, when the observation is quantitative.
    """


@dataclass
class HasParameterRange(PropertyDescriptor):
    """
    The value of a quantitative parameter.
    """


@dataclass
class HasPrediction(PropertyDescriptor):
    """
    A sensory consequence expected at the event, which the control point checks.
    """


@dataclass
class HasPredictionError(PropertyDescriptor):
    """
    An error a mismatching comparison is classified as.
    """


@dataclass
class HasResistingForce(PropertyDescriptor):
    """
    The force that prevails until the event; the event's threshold is its limit.
    """


@dataclass
class HasResponseLatency(PropertyDescriptor):
    """
    The delay before the response changes the motor output: about 100 ms for tactile, over 200 ms for visual triggers in humans.
    """


@dataclass
class HasSignalFeature(PropertyDescriptor):
    """
    The sensory feature whose evolution is described.
    """


@dataclass
class HasTolerance(PropertyDescriptor):
    """
    The admissible deviation from the predicted magnitude or trajectory.
    """


@dataclass
class HoldsEstimate(PropertyDescriptor):
    """
    An estimate stored as content of the representation.
    """


@dataclass
class InitiatedBy(PropertyDescriptor):
    """
    An event that begins the phase. The first phase of a structure has none.
    """


@dataclass
class IntendedEffect(PropertyDescriptor):
    """
    A condition the correction is meant to establish or preserve.
    """


@dataclass
class InvokesCorrection(PropertyDescriptor):
    """
    A corrective response a mismatch releases.
    """


@dataclass
class InvolvesRole(PropertyDescriptor):
    """
    A participant role taking part in the phase.
    """


@dataclass
class MarginOn(PropertyDescriptor):
    """
    The motor setpoint carrying the safety margin.
    """


@dataclass
class MarginOver(PropertyDescriptor):
    """
    The threshold below the setpoint.
    """


@dataclass
class ObservedAt(PropertyDescriptor):
    """
    The interval in which the observation occurs.
    """


@dataclass
class ObservedOnRole(PropertyDescriptor):
    """
    A participant role involved in the observation.
    """


@dataclass
class ObservesEvent(PropertyDescriptor):
    """
    The observed event.
    """


@dataclass
class ObservesFeature(PropertyDescriptor):
    """
    The observed feature.
    """


@dataclass
class ParameterOnRole(PropertyDescriptor):
    """
    A participant role to which the parameter applies.
    """


@dataclass
class PermittedBy(PropertyDescriptor):
    """
    A successful comparison required in addition to the event gates.
    """


@dataclass
class PredictsEvent(PropertyDescriptor):
    """
    The expected event.
    """


@dataclass
class PredictsFeature(PropertyDescriptor):
    """
    The expected signal feature.
    """


@dataclass
class PredictsSensoryEvent(PropertyDescriptor):
    """
    A sensory event the controller predicts, in particular those that signify attainment of its subgoal.
    """


@dataclass
class PredictsTrajectory(PropertyDescriptor):
    """
    The structured feature change expected during the phase.
    """


@dataclass
class ProducesModelUpdate(PropertyDescriptor):
    """
    A representation update a mismatch supports.
    """


@dataclass
class RateOf(PropertyDescriptor):
    """
    The force setpoint whose rate of change is commanded.
    """


@dataclass
class ResolvesOpposition(PropertyDescriptor):
    """
    The force opposition whose driving force overcomes its resisting force at the event.
    """


@dataclass
class ScalesSetpoint(PropertyDescriptor):
    """
    A motor setpoint the estimate scales feedforward, before any sensory confirmation is available.
    """


@dataclass
class SignalsSubgoal(PropertyDescriptor):
    """
    The subgoal whose attainment the predicted sensory event signifies.
    """


@dataclass
class StartsForce(PropertyDescriptor):
    """
    A force that begins to act at the event, such as the support reaction at touchdown.
    """


@dataclass
class SuppliesStateEstimate(PropertyDescriptor):
    """
    The next phase's initial state supplied by a successful comparison.
    """


@dataclass
class TargetsCondition(PropertyDescriptor):
    """
    The condition the phase is meant to bring about.
    """


@dataclass
class TargetsSetpoint(PropertyDescriptor):
    """
    A motor setpoint the response changes.
    """


@dataclass
class TerminatedBy(PropertyDescriptor):
    """
    An event that ends the phase. A phase without a boundary is not a phase.
    """


@dataclass
class TerminatesStateType(PropertyDescriptor):
    """
    A condition that stops holding when an event of this type occurs. The counterpart of aicor:initiatesStateType.
    """


@dataclass
class ToPhase(PropertyDescriptor):
    """
    The phase control enters.
    """


@dataclass
class TriggeredByError(PropertyDescriptor):
    """
    A prediction error that releases the response or supplies evidence for the update.
    """


@dataclass
class UpdatesEstimate(PropertyDescriptor):
    """
    An estimate the revision changes.
    """


@dataclass
class UpdatesRepresentation(PropertyDescriptor):
    """
    The representation in which the content is changed.
    """


@dataclass
class ViolationSignalledBy(PropertyDescriptor):
    """
    The failure event that registers the condition ceasing to hold.
    """


@dataclass
class BetweenRoleFillers(PropertyDescriptor):
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#betweenRoleFillers.
    """


@dataclass
class GeneratedBy(PropertyDescriptor):
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#generatedBy.
    """


@dataclass
class HasConditionType(PropertyDescriptor):
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#hasConditionType.
    """


@dataclass
class InitiatesStateType(PropertyDescriptor):
    """
    The linked ontology property http://aicor.knowledge/l2-ontology.owl#initiatesStateType.
    """


# %% descriptor attachments

ForceComponent.acts_on_role = ActsOnRole(ForceComponent, "acts_on_role")
ActionPhase.aims_for = AimsFor(ActionPhase, "aims_for")
SensorimotorControlPoint.at_event_type = AtEventType(
    SensorimotorControlPoint, "at_event_type"
)
Subgoal.attainment_signalled_by = AttainmentSignalledBy(
    Subgoal, "attainment_signalled_by"
)
SetpointAttainmentEventType.attains_setpoint = AttainsSetpoint(
    SetpointAttainmentEventType, "attains_setpoint"
)
InitialStateEstimate.based_on_observation = BasedOnObservation(
    InitialStateEstimate, "based_on_observation"
)
SensoryEventPrediction.caused_by_event = CausedByEvent(
    SensoryEventPrediction, "caused_by_event"
)
ContinuousSignalPrediction.compared_with = ComparedWith(
    ContinuousSignalPrediction, "compared_with"
)
PredictionComparison.compares_observation = ComparesObservation(
    PredictionComparison, "compares_observation"
)
PredictionComparison.compares_prediction = ComparesPrediction(
    PredictionComparison, "compares_prediction"
)
PhaseConditionType.condition_on_role = ConditionOnRole(
    PhaseConditionType, "condition_on_role"
)
PredictionComparison.confirms_subgoal = ConfirmsSubgoal(
    PredictionComparison, "confirms_subgoal"
)
ActionPhase.controlled_by = ControlledBy(ActionPhase, "controlled_by")
ContinuousSignalPrediction.controls_setpoint = ControlsSetpoint(
    ContinuousSignalPrediction, "controls_setpoint"
)
CorrectiveAction.correction_terminated_by = CorrectionTerminatedBy(
    CorrectiveAction, "correction_terminated_by"
)
ThresholdCrossingEventType.crosses_threshold = CrossesThreshold(
    ThresholdCrossingEventType, "crosses_threshold"
)
ActionPhaseStructure.defines_comparison = DefinesComparison(
    ActionPhaseStructure, "defines_comparison"
)
ActionPhaseStructure.defines_component = DefinesComponent(
    ActionPhaseStructure, "defines_component"
)
ActionPhaseStructure.defines_condition_type = DefinesConditionType(
    ActionPhaseStructure, "defines_condition_type"
)
ActionPhaseStructure.defines_continuous_prediction = DefinesContinuousPrediction(
    ActionPhaseStructure, "defines_continuous_prediction"
)
ActionPhaseStructure.defines_control_point = DefinesControlPoint(
    ActionPhaseStructure, "defines_control_point"
)
ActionPhaseStructure.defines_control_point_exemption = DefinesControlPointExemption(
    ActionPhaseStructure, "defines_control_point_exemption"
)
ActionPhaseStructure.defines_controller = DefinesController(
    ActionPhaseStructure, "defines_controller"
)
ActionPhaseStructure.defines_correction = DefinesCorrection(
    ActionPhaseStructure, "defines_correction"
)
ActionPhaseStructure.defines_event_type = DefinesEventType(
    ActionPhaseStructure, "defines_event_type"
)
ActionPhaseStructure.defines_force_opposition = DefinesForceOpposition(
    ActionPhaseStructure, "defines_force_opposition"
)
ActionPhaseStructure.defines_initial_state_estimate = DefinesInitialStateEstimate(
    ActionPhaseStructure, "defines_initial_state_estimate"
)
ActionPhaseStructure.defines_model_update = DefinesModelUpdate(
    ActionPhaseStructure, "defines_model_update"
)
ActionPhaseStructure.defines_observation_type = DefinesObservationType(
    ActionPhaseStructure, "defines_observation_type"
)
ActionPhaseStructure.defines_parameter = DefinesParameter(
    ActionPhaseStructure, "defines_parameter"
)
ActionPhaseStructure.defines_phase = DefinesPhase(ActionPhaseStructure, "defines_phase")
ActionPhaseStructure.defines_prediction = DefinesPrediction(
    ActionPhaseStructure, "defines_prediction"
)
ActionPhaseStructure.defines_prediction_error = DefinesPredictionError(
    ActionPhaseStructure, "defines_prediction_error"
)
ActionPhaseStructure.defines_role = DefinesRole(ActionPhaseStructure, "defines_role")
ActionPhaseStructure.defines_sensory_feature = DefinesSensoryFeature(
    ActionPhaseStructure, "defines_sensory_feature"
)
ActionPhaseStructure.defines_subgoal = DefinesSubgoal(
    ActionPhaseStructure, "defines_subgoal"
)
ActionPhaseStructure.defines_transition = DefinesTransition(
    ActionPhaseStructure, "defines_transition"
)
MechanicalEventType.ends_force = EndsForce(MechanicalEventType, "ends_force")
InitialStateEstimate.estimates_condition = EstimatesCondition(
    InitialStateEstimate, "estimates_condition"
)
BoundaryControlExemption.exempts_event = ExemptsEvent(
    BoundaryControlExemption, "exempts_event"
)
ForceComponent.exerted_by_role = ExertedByRole(ForceComponent, "exerted_by_role")
SensoryEventPrediction.expected_on_role = ExpectedOnRole(
    SensoryEventPrediction, "expected_on_role"
)
InitialStateEstimate.for_phase = ForPhase(InitialStateEstimate, "for_phase")
PhaseTransition.from_phase = FromPhase(PhaseTransition, "from_phase")
PhaseTransition.gated_by = GatedBy(PhaseTransition, "gated_by")
ActionPhaseController.governed_by = GovernedBy(ActionPhaseController, "governed_by")
ActionPhase.has_active_parameter = HasActiveParameter(
    ActionPhase, "has_active_parameter"
)
Parameter.has_categorical_value = HasCategoricalValue(
    Parameter, "has_categorical_value"
)
SensorimotorControlPoint.has_comparison = HasComparison(
    SensorimotorControlPoint, "has_comparison"
)
ActionPhase.has_condition_requirement = HasConditionRequirement(
    ActionPhase, "has_condition_requirement"
)
ActionPhaseController.has_continuous_prediction = HasContinuousPrediction(
    ActionPhaseController, "has_continuous_prediction"
)
ActionPhaseController.has_corrective_program = HasCorrectiveProgram(
    ActionPhaseController, "has_corrective_program"
)
PredictionError.has_discrepancy = HasDiscrepancy(PredictionError, "has_discrepancy")
ForceOpposition.has_driving_force = HasDrivingForce(
    ForceOpposition, "has_driving_force"
)
SensoryEventPrediction.has_expected_magnitude = HasExpectedMagnitude(
    SensoryEventPrediction, "has_expected_magnitude"
)
SignalTrajectory.has_expected_range = HasExpectedRange(
    SignalTrajectory, "has_expected_range"
)
SensoryEventPrediction.has_expected_timing = HasExpectedTiming(
    SensoryEventPrediction, "has_expected_timing"
)
SensoryObservationType.has_observed_magnitude = HasObservedMagnitude(
    SensoryObservationType, "has_observed_magnitude"
)
Parameter.has_parameter_range = HasParameterRange(Parameter, "has_parameter_range")
SensorimotorControlPoint.has_prediction = HasPrediction(
    SensorimotorControlPoint, "has_prediction"
)
PredictionComparison.has_prediction_error = HasPredictionError(
    PredictionComparison, "has_prediction_error"
)
ForceOpposition.has_resisting_force = HasResistingForce(
    ForceOpposition, "has_resisting_force"
)
CorrectiveAction.has_response_latency = HasResponseLatency(
    CorrectiveAction, "has_response_latency"
)
SignalTrajectory.has_signal_feature = HasSignalFeature(
    SignalTrajectory, "has_signal_feature"
)
ContinuousSignalPrediction.has_tolerance = HasTolerance(
    ContinuousSignalPrediction, "has_tolerance"
)
EstimateRepresentation.holds_estimate = HoldsEstimate(
    EstimateRepresentation, "holds_estimate"
)
ActionPhase.initiated_by = InitiatedBy(ActionPhase, "initiated_by")
CorrectiveAction.intended_effect = IntendedEffect(CorrectiveAction, "intended_effect")
PredictionComparison.invokes_correction = InvokesCorrection(
    PredictionComparison, "invokes_correction"
)
ActionPhase.involves_role = InvolvesRole(ActionPhase, "involves_role")
SafetyMargin.margin_on = MarginOn(SafetyMargin, "margin_on")
SafetyMargin.margin_over = MarginOver(SafetyMargin, "margin_over")
SensoryObservationType.observed_at = ObservedAt(SensoryObservationType, "observed_at")
SensoryObservationType.observed_on_role = ObservedOnRole(
    SensoryObservationType, "observed_on_role"
)
SensoryObservationType.observes_event = ObservesEvent(
    SensoryObservationType, "observes_event"
)
SensoryObservationType.observes_feature = ObservesFeature(
    SensoryObservationType, "observes_feature"
)
Parameter.parameter_on_role = ParameterOnRole(Parameter, "parameter_on_role")
PhaseTransition.permitted_by = PermittedBy(PhaseTransition, "permitted_by")
SensoryEventPrediction.predicts_event = PredictsEvent(
    SensoryEventPrediction, "predicts_event"
)
SensoryEventPrediction.predicts_feature = PredictsFeature(
    SensoryEventPrediction, "predicts_feature"
)
ActionPhaseController.predicts_sensory_event = PredictsSensoryEvent(
    ActionPhaseController, "predicts_sensory_event"
)
ContinuousSignalPrediction.predicts_trajectory = PredictsTrajectory(
    ContinuousSignalPrediction, "predicts_trajectory"
)
PredictionComparison.produces_model_update = ProducesModelUpdate(
    PredictionComparison, "produces_model_update"
)
ForceRateSetpoint.rate_of = RateOf(ForceRateSetpoint, "rate_of")
MechanicalEventType.resolves_opposition = ResolvesOpposition(
    MechanicalEventType, "resolves_opposition"
)
Estimate.scales_setpoint = ScalesSetpoint(Estimate, "scales_setpoint")
SensoryEventPrediction.signals_subgoal = SignalsSubgoal(
    SensoryEventPrediction, "signals_subgoal"
)
MechanicalEventType.starts_force = StartsForce(MechanicalEventType, "starts_force")
PredictionComparison.supplies_state_estimate = SuppliesStateEstimate(
    PredictionComparison, "supplies_state_estimate"
)
Subgoal.targets_condition = TargetsCondition(Subgoal, "targets_condition")
CorrectiveAction.targets_setpoint = TargetsSetpoint(
    CorrectiveAction, "targets_setpoint"
)
ActionPhase.terminated_by = TerminatedBy(ActionPhase, "terminated_by")
PhaseBoundaryEventType.terminates_state_type = TerminatesStateType(
    PhaseBoundaryEventType, "terminates_state_type"
)
PhaseTransition.to_phase = ToPhase(PhaseTransition, "to_phase")
CorrectiveAction.triggered_by_error = TriggeredByError(
    CorrectiveAction, "triggered_by_error"
)
ModelUpdate.triggered_by_error = TriggeredByError(ModelUpdate, "triggered_by_error")
ModelUpdate.updates_estimate = UpdatesEstimate(ModelUpdate, "updates_estimate")
ModelUpdate.updates_representation = UpdatesRepresentation(
    ModelUpdate, "updates_representation"
)
ConditionRequirement.violation_signalled_by = ViolationSignalledBy(
    ConditionRequirement, "violation_signalled_by"
)
MechanicalEventType.between_role_fillers = BetweenRoleFillers(
    MechanicalEventType, "between_role_fillers"
)
SetpointAttainmentEventType.between_role_fillers = BetweenRoleFillers(
    SetpointAttainmentEventType, "between_role_fillers"
)
ContinuousSignalPrediction.generated_by = GeneratedBy(
    ContinuousSignalPrediction, "generated_by"
)
SensoryEventPrediction.generated_by = GeneratedBy(
    SensoryEventPrediction, "generated_by"
)
ConditionRequirement.has_condition_type = HasConditionType(
    ConditionRequirement, "has_condition_type"
)
PhaseBoundaryEventType.initiates_state_type = InitiatesStateType(
    PhaseBoundaryEventType, "initiates_state_type"
)
