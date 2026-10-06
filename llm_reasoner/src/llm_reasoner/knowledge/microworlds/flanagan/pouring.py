"""
Substance-transfer vocabulary extending prehensile manipulation.

This module owns flow, pouring roles, material regimes, transfer variables, and pouring-
specific parameters. General contact, grasp, lift, transport, and release concepts are
owned by :mod:`flanagan.prehension`; general sensory prediction is owned by
:mod:`flanagan.core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    ContinuousSignalPrediction,
    DynamicsEstimate,
    ManipulationRole,
    MechanicalEventType,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    ParameterValueForm,
    PhysicalDimension,
    SetpointAttained,
    TaskSetpoint,
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge.microworlds.flanagan.namespaces import FlanaganNamespace
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    PrehensileManipulandum,
    PrehensionPhaseStructure,
)
from llm_reasoner.knowledge.ontological.decorators import (
    ontological,
    prop,
    value_partition,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel, Rigidity

POUR = FlanaganNamespace.POURING
"""
Namespace for substance-transfer concepts.
"""

SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"
"""
SOMA namespace used for external alignments.
"""


@value_partition(POUR + "MaterialRegime")
class MaterialRegime(StrEnum):
    """
    The physical flow behavior governing a transferred substance.
    """

    LIQUID = "liquid"
    """
    A low-shear material that flows and may slosh.
    """

    GRANULAR = "granular"
    """
    Discrete particles whose bulk flow is not characterized by viscosity.
    """

    COHESIVE = "cohesive"
    """
    A highly viscous or cohesive material with delayed flow.
    """


@value_partition(POUR + "TransferTerminationReason")
class TransferTerminationReason(StrEnum):
    """
    The condition intentionally or exceptionally ending transfer.
    """

    TARGET_QUANTITY = "target_quantity"
    """
    The requested transferred quantity has been reached.
    """

    SOURCE_EMPTY = "source_empty"
    """
    No transferable material remains in the source.
    """

    EXPLICIT_INSTRUCTION = "explicit_instruction"
    """
    An external instruction requests transfer cessation.
    """

    FAILURE = "failure"
    """
    A spill or other failure forces transfer cessation.
    """


# %% substance-transfer apparatus
# Genuinely about pouring.

# %% substance-transfer roles


@ontological(
    iri=POUR + "SourceContainer",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
    aligns_to=SOMA + "Container",
)
@dataclass(eq=False)
class SourceContainer(PrehensileManipulandum):
    """
    The vessel that is grasped and tilted.

    It is the manipulandum, but unlike a rigid manipulandum its inertial properties
    change during the action -- which is the distinctive demand pouring places on the
    framework.
    """


@ontological(
    iri=POUR + "TargetReceptacle",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
    aligns_to=SOMA + "Destination",
)
@dataclass(eq=False)
class TargetReceptacle(ManipulationRole):
    """
    The vessel or surface receiving the substance.
    """


@ontological(
    iri=POUR + "PouredSubstance",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
    aligns_to=SOMA + "PouredObject",
)
@dataclass(eq=False)
class PouredSubstance(ManipulationRole):
    """
    The substance transferred.

    Distinct from the source container: the substance is what the task is about,
    the container is what is manipulated.
    """


@ontological(
    iri=POUR + "PouringLip",
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker="Symbol",
)
@dataclass(eq=False)
class PouringLip(ManipulationRole):
    """
    The edge over which the substance leaves the source container.

    The geometric locus of the flow-onset event.
    """


# %% flow events


@ontological(iri=POUR + "FlowEventType", disjoint_siblings=True)
@dataclass(eq=False)
class FlowEventType(MechanicalEventType):
    """
    An event constituted by a change in substance motion.

    Flow identity is orthogonal to threshold crossing. A flow event may therefore also
    be a :class:`ThresholdCrossingEventType`.
    """


@ontological(iri=POUR + "FlowOnset")
@dataclass(eq=False)
class FlowOnset(FlowEventType, ThresholdCrossingEventType):
    """
    Substance first crosses the lip after the onset threshold is crossed.
    """


@ontological(iri=POUR + "FlowCessation")
@dataclass(eq=False)
class FlowCessation(FlowEventType):
    """
    Substance stops crossing the pouring lip.
    """


@ontological(iri=POUR + "SpillEvent")
@dataclass(eq=False)
class SpillEvent(FlowEventType, ThresholdCrossingEventType):
    """
    Substance flow exceeds the containment threshold outside the receptacle.
    """


@ontological(iri=POUR + "TargetQuantityReached")
@dataclass(eq=False)
class TargetQuantityReached(SetpointAttained):
    """
    Transferred mass or volume meets the requested task quantity.
    """


@ontological(iri=POUR + "SourceEmpty")
@dataclass(eq=False)
class SourceEmpty(SetpointAttained):
    """
    The remaining transferable quantity in the source reaches zero.
    """


@ontological(iri=POUR + "LipAligned")
@dataclass(eq=False)
class LipAligned(SetpointAttained):
    """
    Commanded pose placing the lip over the receptacle opening has been reached.
    """


@ontological(iri=POUR + "UprightRestored")
@dataclass(eq=False)
class UprightRestored(SetpointAttained):
    """
    The container has been returned to a stable orientation.
    """


# %% substance-transfer phases


@ontological(iri=POUR + "AlignmentPhase", display_name="Align")
@dataclass(eq=False)
class AlignmentPhase(ActionPhase):
    """
    Positioning of the lip over the receptacle prior to tilting.
    """


@ontological(iri=POUR + "TiltToFlowPhase", display_name="Tilt")
@dataclass(eq=False)
class TiltToFlowPhase(ActionPhase):
    """
    Rotation of the source container until the substance reaches the lip and flow
    begins.

    Ends at flow onset.
    """


@ontological(iri=POUR + "FlowRegulationPhase", display_name="Pour")
@dataclass(eq=False)
class FlowRegulationPhase(ActionPhase):
    """
    Continuous modulation of tilt angle and grip/load force while substance leaves the
    container, against tracked volume in the receptacle.

    The one phase that is COUPLED and CONTINUOUSLY MODULATED at once -- exactly the
    combination the orthogonal regime enums exist for, and the case a sibling taxonomy
    could not express. It is also where the discrete event-boundary assumption of the
    core framework is weakest and continuous feedback dominates, so instances must
    monitor at least one signal.
    """


@ontological(iri=POUR + "FlowTerminationPhase", display_name="StopFlow")
@dataclass(eq=False)
class FlowTerminationPhase(ActionPhase):
    """
    Counter-rotation to bring the substance back below the lip.

    Ends at flow cessation.
    """


@ontological(iri=POUR + "ReturnUprightPhase", display_name="Upright")
@dataclass(eq=False)
class ReturnUprightPhase(ActionPhase):
    """
    Restoration of the container to a stable orientation after flow has stopped.
    """


# %% pouring parameters


@ontological(iri=POUR + "DynamicViscosityEstimate")
@dataclass(eq=False)
class DynamicViscosityEstimate(ObjectPropertyEstimate):
    """
    Anticipated dynamic viscosity of a liquid or cohesive substance.
    """

    required_dimension: ClassVar[PhysicalDimension] = (
        PhysicalDimension.DYNAMIC_VISCOSITY
    )
    """
    Dynamic viscosity requires the dynamic-viscosity dimension.
    """


@ontological(iri=POUR + "RemainingSourceQuantityEstimate")
@dataclass(eq=False)
class RemainingSourceQuantityEstimate(ObjectStateEstimate):
    """
    Anticipated transferable mass or volume remaining in the source.
    """

    permitted_dimensions: ClassVar[frozenset[PhysicalDimension]] = frozenset(
        {PhysicalDimension.MASS, PhysicalDimension.VOLUME}
    )
    """
    Remaining quantity may be represented by mass or volume.
    """


@ontological(iri=POUR + "TargetFillStateEstimate")
@dataclass(eq=False)
class TargetFillStateEstimate(ObjectStateEstimate):
    """
    Anticipated filled fraction of the target receptacle.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.DIMENSIONLESS
    """
    Target fill state is a dimensionless fraction.
    """


@ontological(iri=POUR + "AlignmentErrorEstimate")
@dataclass(eq=False)
class AlignmentErrorEstimate(ObjectStateEstimate):
    """
    Angular deviation between the pouring lip and receptacle opening.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.ANGLE
    """
    Alignment error requires the angle dimension.
    """


@ontological(iri=POUR + "SloshDynamicsEstimate")
@dataclass(eq=False)
class SloshDynamicsEstimate(DynamicsEstimate):
    """
    Categorical model of internal liquid motion under acceleration.
    """

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """
    Slosh dynamics requires a categorical value.
    """


@ontological(iri=POUR + "GranularFlowDynamicsEstimate")
@dataclass(eq=False)
class GranularFlowDynamicsEstimate(DynamicsEstimate):
    """
    Categorical model of particle flow, jamming, and avalanching.
    """

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """
    Granular flow dynamics requires a categorical value.
    """


@ontological(iri=POUR + "CohesiveFlowDynamicsEstimate")
@dataclass(eq=False)
class CohesiveFlowDynamicsEstimate(DynamicsEstimate):
    """
    Categorical model of delayed or intermittent cohesive flow.
    """

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """
    Cohesive flow dynamics requires a categorical value.
    """


@ontological(iri=POUR + "CenterOfMassShiftEstimate")
@dataclass(eq=False)
class CenterOfMassShiftEstimate(DynamicsEstimate):
    """
    Predicted centre-of-mass displacement as substance leaves.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.LENGTH
    """
    Centre-of-mass displacement requires the length dimension.
    """


@ontological(iri=POUR + "TiltAngleSetpoint")
@dataclass(eq=False)
class TiltAngleSetpoint(MotorSetpoint):
    """
    Commanded source-container orientation about its pouring axis.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.ANGLE
    """
    Tilt angle requires the angle dimension.
    """


@ontological(iri=POUR + "TiltRateSetpoint")
@dataclass(eq=False)
class TiltRateSetpoint(MotorSetpoint):
    """
    Commanded angular velocity of source-container tilting.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.ANGULAR_RATE
    """
    Tilt rate requires the angular-rate dimension.
    """


@ontological(iri=POUR + "FlowRateSetpoint")
@dataclass(eq=False)
class FlowRateSetpoint(TaskSetpoint):
    """
    Desired mass or volume transferred per unit time.
    """

    permitted_dimensions: ClassVar[frozenset[PhysicalDimension]] = frozenset(
        {
            PhysicalDimension.MASS_FLOW_RATE,
            PhysicalDimension.VOLUMETRIC_FLOW_RATE,
        }
    )
    """
    Flow rate may be represented by mass rate or volumetric rate.
    """


@ontological(iri=POUR + "TorqueSetpoint")
@dataclass(eq=False)
class TorqueSetpoint(MotorSetpoint):
    """
    Commanded moment about the grasp.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.TORQUE
    """
    Torque requires the torque dimension.
    """


@ontological(iri=POUR + "PourPoseSetpoint")
@dataclass(eq=False)
class PourPoseSetpoint(MotorSetpoint):
    """
    Categorical reference to the desired pouring pose.
    """

    required_value_form: ClassVar[ParameterValueForm] = ParameterValueForm.CATEGORICAL
    """
    Pour pose requires a categorical value.
    """


@ontological(iri=POUR + "LipAlignmentSetpoint")
@dataclass(eq=False)
class LipAlignmentSetpoint(MotorSetpoint):
    """
    Commanded angular alignment of the pouring lip with the receptacle.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.ANGLE
    """
    Lip alignment requires the angle dimension.
    """


@ontological(iri=POUR + "UprightAngleSetpoint")
@dataclass(eq=False)
class UprightAngleSetpoint(MotorSetpoint):
    """
    Commanded upright orientation of the source container.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.ANGLE
    """
    Upright orientation requires the angle dimension.
    """


@ontological(iri=POUR + "TransferredQuantitySetpoint")
@dataclass(eq=False)
class TransferredQuantitySetpoint(TaskSetpoint):
    """
    Desired mass or volume of substance to transfer.
    """

    permitted_dimensions: ClassVar[frozenset[PhysicalDimension]] = frozenset(
        {PhysicalDimension.MASS, PhysicalDimension.VOLUME}
    )
    """
    Transferred quantity may be represented by mass or volume.
    """


@ontological(iri=POUR + "FlowOnsetThreshold")
@dataclass(eq=False)
class FlowOnsetThreshold(MechanicalThreshold):
    """
    Tilt angle at which the substance begins crossing the pouring lip.
    """

    required_dimension: ClassVar[PhysicalDimension] = PhysicalDimension.ANGLE
    """
    Flow-onset threshold requires the angle dimension.
    """


@ontological(iri=POUR + "SpillThreshold")
@dataclass(eq=False)
class SpillThreshold(MechanicalThreshold):
    """
    Flow-rate threshold beyond which transfer containment is violated.
    """

    permitted_dimensions: ClassVar[frozenset[PhysicalDimension]] = frozenset(
        {
            PhysicalDimension.MASS_FLOW_RATE,
            PhysicalDimension.VOLUMETRIC_FLOW_RATE,
        }
    )
    """
    A spill threshold may use mass rate or volumetric rate.
    """


# %% structure


def continuous_control_contract_description() -> str:
    """
    Return the required links for continuous pouring control.
    """
    return (
        "Every CONTINUOUS_FEEDBACK or HYBRID phase must occur in "
        "ContinuousSignalPrediction.predicted_for_phase; that prediction's signal "
        "must also occur in phase.monitors_signal; predicted_trajectory must be "
        "present; controlled_variable must be a Setpoint listed in "
        "structure.defines_parameter."
    )


@ontological(iri=POUR + "PouringPhaseStructure")
@dataclass(eq=False)
class PouringPhaseStructure(PrehensionPhaseStructure):
    """
    A substance-transfer phase structure composed with prehension.
    """

    material_regime: MaterialRegime | None = prop(
        functional=True,
        min_count=1,
        iri=POUR + "hasMaterialRegime",
    )
    """
    The flow behavior governing the transferred substance.
    """

    termination_reason: TransferTerminationReason | None = prop(
        functional=True,
        min_count=1,
        iri=POUR + "hasTerminationReason",
    )
    """
    The condition that ends transfer in this task variant.
    """

    has_continuous_prediction: list[ContinuousSignalPrediction] = prop(
        default_factory=list,
        iri=POUR + "hasContinuousPrediction",
        comment=continuous_control_contract_description(),
    )
    """
    Continuous predictions that close the feedback loops of pouring phases.
    """
