"""
How grip force is controlled: scaled to the load, corrected after errors and adjusted
ahead of impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    CorrectiveAction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    SafetyMargin,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CorrectionDirection,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    LoadForceEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    PREHENSION,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.parameters import (
    ExpectedFrictionCoefficient,
    GripForceSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.phases import (
    ReplacePhase,
)
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% responses to error


@PREHENSION.concept()
@dataclass(eq=False)
class CorrectiveGripResponse(CorrectiveAction):
    """
    A prediction-error response that changes a grip-force target.
    """

    targets_setpoint: list[GripForceSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("targets_setpoint")
    )
    """
    The grip-force targets changed by the response.
    """


# %% anticipation


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING, claims=("FJ-009",))
@dataclass(eq=False)
class AnticipatoryGripAdjustment(KnowledgeEntity):
    """
    A feedforward grip change scheduled from a predicted touchdown while lowering.
    """

    triggered_by_prediction: Optional[SensoryEventPrediction] = relation(
        functional=True, minimum_count=1
    )
    """
    The predicted event consequence scheduling the adjustment.
    """

    targets_grip_force: Optional[GripForceSetpoint] = relation(
        functional=True, minimum_count=1
    )
    """
    The grip-force target changed before the predicted event.
    """

    adjustment_direction: Optional[CorrectionDirection] = attribute(
        functional=True, minimum_count=1
    )
    """
    The direction in which grip force changes.
    """

    adjustment_active_in: list[ReplacePhase] = relation(minimum_count=1)
    """
    The lowering phases during which the anticipatory change is active.
    """


# %% coupling


@PREHENSION.concept(
    alignments=(subclass_of(DUL, "Relation"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-008", "FJ-025"),
)
@dataclass(eq=False)
class GripLoadCoupling(KnowledgeEntity):
    """
    A functional grip response derived from load, friction, margin, and timing.
    """

    predicted_load: Optional[LoadForceEstimate] = relation(
        functional=True, minimum_count=1
    )
    """
    The represented tangential load for which grip force is scaled.
    """

    friction_estimate: Optional[ExpectedFrictionCoefficient] = relation(
        functional=True, minimum_count=1
    )
    """
    The estimated grasp-interface friction limiting tangential force.
    """

    safety_margin: Optional[SafetyMargin] = relation(functional=True, minimum_count=1)
    """
    The additional grip force maintained above the slip threshold.
    """

    grip_force_response: Optional[GripForceSetpoint] = relation(
        functional=True, minimum_count=1
    )
    """
    The grip-force target produced by the functional relationship.
    """

    temporal_coupling: Optional[TemporalWindow] = relation(
        functional=True, minimum_count=1
    )
    """
    The admissible timing relation between predicted load and grip response.
    """

    coupling_active_in: list[ActionPhase] = relation()
    """
    The phases in which grip and load remain functionally coupled.
    """
