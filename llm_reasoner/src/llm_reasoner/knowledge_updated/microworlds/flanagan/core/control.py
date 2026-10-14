"""
Comparing prediction with evidence, and what follows from the result.

At a sensorimotor control point a prediction is compared with an observation. A match
confirms the subgoal, supplies the next phase's initial state and permits the transition.
A mismatch is classified into errors, which release corrections and model updates. A
correction protects the current action and a model update changes future predictions, so
the two are kept apart.

A transition names the comparisons that permit it. The comparison does not name the
transition back, so the link is stated once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    SOMA,
    related_to,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    PhaseConditionType,
    Subgoal,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    EstimateRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Estimate,
    MotorSetpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    SensoryEventPrediction,
    SensoryObservationType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    MagnitudeRange,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ComparisonOutcome,
    CorrectionDirection,
    EstimateUpdateDirection,
    PredictionErrorKind,
    RecoveryMode,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import (
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% errors and their consequences


@CORE.concept(
    alignments=(subclass_of(DUL, "Situation"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-005",),
)
@dataclass(eq=False)
class PredictionError(KnowledgeEntity):
    """
    A classified discrepancy between a prediction and an observation.
    """

    error_kind: Optional[PredictionErrorKind] = attribute(
        functional=True, minimum_count=1
    )
    """
    The comparison dimension along which the prediction failed.
    """

    discrepancy: Optional[MagnitudeRange] = relation(functional=True)
    """
    The discrepancy magnitude, when the error is quantitative.
    """


@CORE.concept(
    alignments=(subclass_of(SOMA, "PhysicalTask"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=(
        "FJ-005",
        "FJ-010",
    ),
)
@dataclass(eq=False)
class CorrectiveAction(KnowledgeEntity):
    """
    A typed motor response to one or more prediction errors.
    """

    triggered_by_error: list[PredictionError] = relation(minimum_count=1)
    """
    The prediction errors that release the response.
    """

    targets_setpoint: list[MotorSetpoint] = relation(minimum_count=1)
    """
    The motor setpoints changed by the response.
    """

    correction_direction: Optional[CorrectionDirection] = attribute(
        functional=True, minimum_count=1
    )
    """
    How the targeted setpoints are changed.
    """

    intended_effect: list[PhaseConditionType] = relation(minimum_count=1)
    """
    The state the correction is intended to establish or preserve.
    """

    recovery_mode: Optional[RecoveryMode] = attribute(functional=True, minimum_count=1)
    """
    How phase execution proceeds after applying the response.
    """


@CORE.concept(
    alignments=(related_to(DUL, "Situation"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-010",),
)
@dataclass(eq=False)
class ModelUpdate(KnowledgeEntity):
    """
    A revision of represented estimates driven by prediction errors.
    """

    triggered_by_error: list[PredictionError] = relation(minimum_count=1)
    """
    The prediction errors supplying evidence for the revision.
    """

    updates_estimate: list[Estimate] = relation(minimum_count=1)
    """
    The estimates changed by the revision.
    """

    updates_representation: Optional[EstimateRepresentation] = relation(
        functional=True, minimum_count=1
    )
    """
    The representation in which the content is changed.
    """

    update_direction: Optional[EstimateUpdateDirection] = attribute(
        functional=True, minimum_count=1
    )
    """
    How the represented estimates change.
    """


# %% comparison


@CORE.concept(
    alignments=(subclass_of(DUL, "Situation"),),
    status=EvidenceStatus.AUTHOR_HYPOTHESIS,
    claims=("FJ-007",),
)
@dataclass(eq=False)
class InitialStateEstimate(KnowledgeEntity):
    """
    The observation-grounded state supplied to the next action phase.
    """

    for_phase: Optional[ActionPhase] = relation(functional=True, minimum_count=1)
    """
    The phase whose initial state is estimated.
    """

    based_on_observation: list[SensoryObservationType] = relation(minimum_count=1)
    """
    The observations supporting the state estimate.
    """

    estimates_condition: list[PhaseConditionType] = relation(minimum_count=1)
    """
    The conditions estimated to hold at phase entry.
    """


@CORE.concept(
    alignments=(subclass_of(DUL, "Situation"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-005",),
)
@dataclass(eq=False)
class PredictionComparison(KnowledgeEntity):
    """
    The operational comparison between one prediction and its observation.
    """

    compares_prediction: Optional[SensoryEventPrediction] = relation(
        functional=True, minimum_count=1
    )
    """
    The sensory prediction being evaluated.
    """

    compares_observation: Optional[SensoryObservationType] = relation(functional=True)
    """
    The received observation, absent only for an absent-event error.
    """

    comparison_outcome: Optional[ComparisonOutcome] = attribute(
        functional=True, minimum_count=1
    )
    """
    Whether the observation matches the prediction.
    """

    confirms_subgoal: Optional[Subgoal] = relation(functional=True)
    """
    The subgoal confirmed by a successful comparison.
    """

    supplies_state_estimate: Optional[InitialStateEstimate] = relation(functional=True)
    """
    The next-phase state supplied by a successful comparison.
    """

    has_error: list[PredictionError] = relation()
    """
    The errors classified by a mismatching comparison.
    """

    invokes_correction: list[CorrectiveAction] = relation()
    """
    The corrective responses released by a mismatch.
    """

    produces_model_update: list[ModelUpdate] = relation()
    """
    The representation updates supported by a mismatch.
    """


@CORE.concept(
    alignments=(subclass_of(DUL, "Relation"),),
    status=EvidenceStatus.AUTHOR_HYPOTHESIS,
    claims=("FJ-001",),
)
@dataclass(eq=False)
class SensorimotorControlPoint(KnowledgeEntity):
    """
    A physical boundary with explicit predictions and comparison outcomes.
    """

    at_event_type: Optional[MechanicalEventType] = relation(
        functional=True, minimum_count=1
    )
    """
    The mechanical event at which comparisons are performed.
    """

    has_prediction: list[SensoryEventPrediction] = relation(minimum_count=1)
    """
    The sensory consequences expected at the event.
    """

    has_comparison: list[PredictionComparison] = relation(minimum_count=1)
    """
    The possible match and mismatch outcomes at the control point.
    """


# %% transitions and exemptions


@CORE.concept(
    alignments=(subclass_of(DUL, "Relation"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
    claims=(
        "FJ-003",
        "FJ-007",
    ),
)
@dataclass(eq=False)
class PhaseTransition(KnowledgeEntity):
    """
    A gated edge of the phase state machine.

    Control passes from one phase to the next when a declared boundary event is
    registered. Physical boundaries preserve the predictive-control semantics, and
    external boundaries support interruption without pretending it is a physical event.
    """

    from_phase: Optional[ActionPhase] = relation(functional=True, minimum_count=1)
    """
    The phase control leaves.
    """

    to_phase: Optional[ActionPhase] = relation(functional=True, minimum_count=1)
    """
    The phase control enters.
    """

    gated_by: list[PhaseBoundaryEventType] = relation(minimum_count=1)
    """
    The boundary events, any of which releases the transition.
    """

    permitted_by: list[PredictionComparison] = relation()
    """
    Successful comparisons required in addition to the event gates.
    """


@CORE.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class BoundaryControlExemption(KnowledgeEntity):
    """
    An explicit reason a phase boundary has no sensorimotor control point.
    """

    exempts_event: Optional[PhaseBoundaryEventType] = relation(
        functional=True, minimum_count=1
    )
    """
    The phase boundary intentionally represented without a control point.
    """

    justification: Optional[ConceptReference] = reference(
        functional=True, minimum_count=1
    )
    """
    The policy, instruction or scope decision justifying the exemption.
    """
