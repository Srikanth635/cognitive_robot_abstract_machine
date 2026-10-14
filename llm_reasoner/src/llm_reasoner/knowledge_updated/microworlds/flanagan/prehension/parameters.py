"""
The setpoints, thresholds and estimates of prehensile control.

Each concept states the form and dimension of the value it holds, so the schema can
offer a model only what suits it and a value of the wrong kind is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    ContactPropertyEstimate,
    ForceRateSetpoint,
    MechanicalThreshold,
    MotorSetpoint,
    ObjectPropertyEstimate,
    ObjectStateEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import PREHENSION
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

QUANTITATIVE = ParameterValueForm.QUANTITATIVE
CATEGORICAL = ParameterValueForm.CATEGORICAL


def quantity(dimension: PhysicalDimension) -> dict[str, object]:
    """
    Return the values fixed for a parameter that is a quantity of a dimension.
    """
    return {"value_form": QUANTITATIVE, "dimension": dimension}


# %% setpoints


@PREHENSION.concept(fixed_values={"value_form": CATEGORICAL})
@dataclass(eq=False)
class ReachPoseSetpoint(MotorSetpoint):
    """
    A categorical target for the pre-contact effector pose.
    """


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.FORCE),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class GripForceSetpoint(MotorSetpoint):
    """
    Normal force target perpendicular to the grasp surface.
    """


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.FORCE),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class LoadForceSetpoint(MotorSetpoint):
    """
    Tangential force target applied to the manipulandum.
    """


@PREHENSION.concept()
@dataclass(eq=False)
class LoadForceRateSetpoint(ForceRateSetpoint):
    """
    The commanded rate of tangential load-force change.
    """


@PREHENSION.concept(fixed_values=quantity(PhysicalDimension.LENGTH))
@dataclass(eq=False)
class LiftClearanceSetpoint(MotorSetpoint):
    """
    The commanded clearance above the initial support.
    """


@PREHENSION.concept(fixed_values={"value_form": CATEGORICAL})
@dataclass(eq=False)
class TransportPoseSetpoint(MotorSetpoint):
    """
    A categorical target for transported manipulandum pose.
    """


@PREHENSION.concept(fixed_values=quantity(PhysicalDimension.LENGTH))
@dataclass(eq=False)
class WithdrawalDistanceSetpoint(MotorSetpoint):
    """
    The commanded post-release effector withdrawal distance.
    """


# %% thresholds


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.FORCE),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class SlipThreshold(MechanicalThreshold):
    """
    The minimum grip force supporting the predicted tangential load.
    """


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.FORCE),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class LiftoffThreshold(MechanicalThreshold):
    """
    The load-force threshold corresponding to manipulandum weight.
    """


@PREHENSION.concept(fixed_values=quantity(PhysicalDimension.FORCE))
@dataclass(eq=False)
class TouchdownThreshold(MechanicalThreshold):
    """
    The support-reaction threshold registering touchdown.
    """


# %% estimates


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.MASS),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-009",),
)
@dataclass(eq=False)
class ExpectedMass(ObjectPropertyEstimate):
    """
    The anticipated manipulandum mass used to predict weight force.
    """


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.DIMENSIONLESS),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-012",),
)
@dataclass(eq=False)
class ExpectedFrictionCoefficient(ContactPropertyEstimate):
    """
    The anticipated friction coefficient at the grasp interface.
    """


@PREHENSION.concept(
    fixed_values=quantity(PhysicalDimension.LENGTH),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-009",),
)
@dataclass(eq=False)
class ExpectedCenterOfMassPosition(ObjectStateEstimate):
    """
    The anticipated centre-of-mass position in the object frame.
    """


@PREHENSION.concept(
    fixed_values={"value_form": CATEGORICAL},
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-009",),
)
@dataclass(eq=False)
class ExpectedShape(ObjectPropertyEstimate):
    """
    The anticipated geometric category used for grasp planning.
    """
