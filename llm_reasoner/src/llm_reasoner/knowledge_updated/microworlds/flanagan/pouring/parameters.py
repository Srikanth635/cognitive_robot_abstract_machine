"""
The setpoints, thresholds and estimates of a substance transfer.

Each concept states the form and dimension of the value it holds, or the dimensions it
permits where a quantity may be measured in more than one way, such as by mass or
volume.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    DynamicsEstimate,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
    TaskSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import POURING
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

QUANTITATIVE = ParameterValueForm.QUANTITATIVE
CATEGORICAL = ParameterValueForm.CATEGORICAL
MASS_OR_VOLUME = frozenset({PhysicalDimension.MASS, PhysicalDimension.VOLUME})
"""
The dimensions in which a quantity of substance is measured.
"""

FLOW_RATES = frozenset(
    {PhysicalDimension.MASS_FLOW_RATE, PhysicalDimension.VOLUMETRIC_FLOW_RATE}
)
"""
The dimensions in which a rate of flow is measured.
"""

ENGINEERING = EvidenceStatus.ENGINEERING_EXTENSION


def quantity(dimension: PhysicalDimension) -> dict[str, object]:
    """
    Return the values fixed for a parameter that is a quantity of a dimension.
    """
    return {"value_form": QUANTITATIVE, "dimension": dimension}


# %% estimates of the substance and the transfer


@POURING.concept(
    fixed_values=quantity(PhysicalDimension.DYNAMIC_VISCOSITY), status=ENGINEERING
)
@dataclass(eq=False)
class DynamicViscosityEstimate(ObjectPropertyEstimate):
    """
    Anticipated dynamic viscosity of a liquid or cohesive substance.
    """


@POURING.concept(
    fixed_values={"value_form": QUANTITATIVE},
    allowed_values={"dimension": MASS_OR_VOLUME},
)
@dataclass(eq=False)
class RemainingSourceQuantityEstimate(ObjectStateEstimate):
    """
    Anticipated transferable mass or volume remaining in the source.
    """


@POURING.concept(
    fixed_values=quantity(PhysicalDimension.DIMENSIONLESS), status=ENGINEERING
)
@dataclass(eq=False)
class TargetFillStateEstimate(ObjectStateEstimate):
    """
    Anticipated filled fraction of the target receptacle.
    """


@POURING.concept(fixed_values=quantity(PhysicalDimension.ANGLE))
@dataclass(eq=False)
class AlignmentErrorEstimate(ObjectStateEstimate):
    """
    Angular deviation between the pouring lip and receptacle opening.
    """


@POURING.concept(fixed_values={"value_form": CATEGORICAL}, status=ENGINEERING)
@dataclass(eq=False)
class SloshDynamicsEstimate(DynamicsEstimate):
    """
    Categorical model of internal liquid motion under acceleration.
    """


@POURING.concept(fixed_values={"value_form": CATEGORICAL})
@dataclass(eq=False)
class GranularFlowDynamicsEstimate(DynamicsEstimate):
    """
    Categorical model of particle flow, jamming, and avalanching.
    """


@POURING.concept(fixed_values={"value_form": CATEGORICAL})
@dataclass(eq=False)
class CohesiveFlowDynamicsEstimate(DynamicsEstimate):
    """
    Categorical model of delayed or intermittent cohesive flow.
    """


@POURING.concept(fixed_values=quantity(PhysicalDimension.LENGTH), status=ENGINEERING)
@dataclass(eq=False)
class CenterOfMassShiftEstimate(DynamicsEstimate):
    """
    Predicted centre-of-mass displacement as substance leaves.
    """


# %% motor setpoints


@POURING.concept(fixed_values=quantity(PhysicalDimension.ANGLE), status=ENGINEERING)
@dataclass(eq=False)
class TiltAngleSetpoint(MotorSetpoint):
    """
    Commanded source-container orientation about its pouring axis.
    """


@POURING.concept(
    fixed_values=quantity(PhysicalDimension.ANGULAR_RATE), status=ENGINEERING
)
@dataclass(eq=False)
class TiltRateSetpoint(MotorSetpoint):
    """
    Commanded angular velocity of source-container tilting.
    """


@POURING.concept(fixed_values=quantity(PhysicalDimension.TORQUE), status=ENGINEERING)
@dataclass(eq=False)
class TorqueSetpoint(MotorSetpoint):
    """
    Commanded moment about the grasp.
    """


@POURING.concept(fixed_values={"value_form": CATEGORICAL})
@dataclass(eq=False)
class PourPoseSetpoint(MotorSetpoint):
    """
    Categorical reference to the desired pouring pose.
    """


@POURING.concept(fixed_values=quantity(PhysicalDimension.ANGLE))
@dataclass(eq=False)
class LipAlignmentSetpoint(MotorSetpoint):
    """
    Commanded angular alignment of the pouring lip with the receptacle.
    """


@POURING.concept(fixed_values=quantity(PhysicalDimension.ANGLE))
@dataclass(eq=False)
class UprightAngleSetpoint(MotorSetpoint):
    """
    Commanded upright orientation of the source container.
    """


# %% task setpoints


@POURING.concept(
    fixed_values={"value_form": QUANTITATIVE}, allowed_values={"dimension": FLOW_RATES}
)
@dataclass(eq=False)
class FlowRateSetpoint(TaskSetpoint):
    """
    Desired mass or volume transferred per unit time.
    """


@POURING.concept(
    fixed_values={"value_form": QUANTITATIVE},
    allowed_values={"dimension": MASS_OR_VOLUME},
    status=ENGINEERING,
)
@dataclass(eq=False)
class TransferredQuantitySetpoint(TaskSetpoint):
    """
    Desired mass or volume of substance to transfer.
    """


# %% thresholds


@POURING.concept(fixed_values=quantity(PhysicalDimension.ANGLE), status=ENGINEERING)
@dataclass(eq=False)
class FlowOnsetThreshold(MechanicalThreshold):
    """
    Tilt angle at which the substance begins crossing the pouring lip.
    """


@POURING.concept(
    fixed_values={"value_form": QUANTITATIVE},
    allowed_values={"dimension": FLOW_RATES},
    status=ENGINEERING,
)
@dataclass(eq=False)
class SpillThreshold(MechanicalThreshold):
    """
    Flow-rate threshold beyond which transfer containment is violated.
    """
