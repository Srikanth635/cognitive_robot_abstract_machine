"""
The quantities a phase structure is parameterised by.

Every parameter says what form of value it holds and in which physical dimension. A
concrete parameter concept states those in its declaration, so the schema can offer the
model only the dimensions that suit it and one check can compare the dimension with the
unit of the value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    CategoricalValue,
    MagnitudeRange,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% every parameter


@CORE.concept(
    alignments=(subclass_of(DUL, "Parameter"),),
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class Parameter(KnowledgeEntity):
    """
    A quantitative or categorical slot in a phase structure.
    """

    value_form: Optional[ParameterValueForm] = attribute(
        functional=True, minimum_count=1
    )
    """
    Whether the parameter holds a magnitude range or a categorical value.
    """

    dimension: Optional[PhysicalDimension] = attribute(functional=True)
    """
    The physical dimension of a quantitative parameter, which the unit of its range must
    measure.
    """

    parameter_range: Optional[MagnitudeRange] = relation(
        functional=True, exclusive_group="value"
    )
    """
    The quantitative value, given exactly when the parameter is quantitative.
    """

    categorical_value: Optional[CategoricalValue] = relation(
        functional=True, exclusive_group="value"
    )
    """
    The categorical value, given exactly when the parameter is categorical.
    """

    parameter_on_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The participant roles to which the parameter applies.
    """


# %% estimates


@CORE.concept(abstract=True, disjoint_siblings=True)
@dataclass(eq=False)
class Estimate(Parameter):
    """
    A belief used to predict or scale manipulation control.
    """

    scales: list[MotorSetpoint] = relation()
    """
    The motor setpoints scaled feedforward by this estimate, before any sensory
    confirmation is available.
    """


@CORE.concept(
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-009",),
)
@dataclass(eq=False)
class ObjectPropertyEstimate(Estimate):
    """
    A belief about a comparatively stable property of an object.
    """


@CORE.concept(abstract=True)
@dataclass(eq=False)
class ContactPropertyEstimate(Estimate):
    """
    A belief about a property of an effector-object contact interface.
    """


@CORE.concept(abstract=True, disjoint_siblings=True)
@dataclass(eq=False)
class ObjectStateEstimate(Estimate):
    """
    A belief about a changeable state of an object.
    """


@CORE.concept(abstract=True, disjoint_siblings=True)
@dataclass(eq=False)
class DynamicsEstimate(Estimate):
    """
    A belief about how an object state evolves under action.
    """


# %% setpoints


@CORE.concept(
    alignments=(subclass_of(SOMA, "Setpoint"),),
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class Setpoint(Parameter):
    """
    A desired value whose attainment can constitute a control event.
    """


@CORE.concept(
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
)
@dataclass(eq=False)
class MotorSetpoint(Setpoint):
    """
    A target directly commanded by the motor controller.
    """


@CORE.concept(
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class TaskSetpoint(Setpoint):
    """
    A desired task outcome rather than a direct motor command.
    """


@CORE.concept(
    fixed_values={
        "value_form": ParameterValueForm.QUANTITATIVE,
        "dimension": PhysicalDimension.FORCE_RATE,
    },
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class ForceRateSetpoint(MotorSetpoint):
    """
    A commanded rate of force change.
    """

    rate_of: Optional[MotorSetpoint] = relation(functional=True, minimum_count=1)
    """
    The force setpoint whose rate is commanded.
    """


# %% thresholds and margins


@CORE.concept(
    alignments=(subclass_of(SOMA, "Threshold"),),
    abstract=True,
    disjoint_siblings=True,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class MechanicalThreshold(Parameter):
    """
    A boundary value whose crossing constitutes a mechanical event.
    """


@CORE.concept(
    fixed_values={"value_form": ParameterValueForm.QUANTITATIVE},
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class SafetyMargin(Parameter):
    """
    The excess of a setpoint over a threshold it must remain clear of.
    """

    margin_on: Optional[MotorSetpoint] = relation(functional=True, minimum_count=1)
    """
    The motor setpoint carrying the safety margin.
    """

    margin_over: Optional[MechanicalThreshold] = relation(
        functional=True, minimum_count=1
    )
    """
    The threshold below the setpoint.
    """
