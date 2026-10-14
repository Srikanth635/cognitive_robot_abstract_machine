"""
The top-level unit of Flanagan knowledge.

A structure is one hypothesis about how a kind of manipulation is organised. It declares
every fragment it is made of, in one collection per kind, and anything else in the
structure may refer only to fragments declared there. That is what makes a hypothesis
self-contained, and so cacheable and versionable as a unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    BoundaryControlExemption,
    CorrectiveAction,
    InitialStateEstimate,
    ModelUpdate,
    PhaseTransition,
    PredictionComparison,
    PredictionError,
    SensorimotorControlPoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    InternalModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    Parameter,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.prediction import (
    ContinuousSignalPrediction,
    SensoryEventPrediction,
    SensoryObservationType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import reference, relation
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@CORE.concept(
    alignments=(subclass_of(DUL, "Workflow"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
    claims=("FJ-001",),
)
@dataclass(eq=False)
class ActionPhaseStructure(KnowledgeEntity):
    """
    The top-level unit of Flanagan knowledge: a phase decomposition of one manipulation
    task.

    One instance is one hypothesis about how a kind of manipulation is organised.
    Deleting it deletes the hypothesis, which is what makes it cacheable and versionable
    as a unit.
    """

    decomposes_task: Optional[ConceptReference] = reference(
        functional=True, minimum_count=1
    )
    """
    The overall task this decomposes, as a reference into the task ontology.

    Exactly one, which is what makes the hypothesis retrievable by action type.
    """

    defines_role: list[RoleSlot] = relation(minimum_count=1, owns=True)
    """
    The participant slots of the task.
    """

    defines_phase: list[ActionPhase] = relation(minimum_count=1, owns=True)
    """
    The phases the task is divided into.
    """

    defines_event_type: list[PhaseBoundaryEventType] = relation(
        minimum_count=1, owns=True
    )
    """
    The boundary events that begin, end and gate the phases.
    """

    defines_transition: list[PhaseTransition] = relation(minimum_count=1, owns=True)
    """
    The directed transitions between phases.
    """

    defines_parameter: list[Parameter] = relation(owns=True)
    """
    The estimates, setpoints, thresholds and margins the phases are parameterised by.
    """

    defines_internal_model: list[InternalModel] = relation(owns=True)
    """
    The internal models and represented content that govern the phases.
    """

    defines_prediction: list[SensoryEventPrediction] = relation(owns=True)
    """
    The sensory predictions expected at physical events.
    """

    defines_observation: list[SensoryObservationType] = relation(owns=True)
    """
    The sensory observation types available for comparison.
    """

    defines_continuous_prediction: list[ContinuousSignalPrediction] = relation(
        owns=True
    )
    """
    The continuous predictions compared throughout phases.
    """

    defines_comparison: list[PredictionComparison] = relation(owns=True)
    """
    The comparisons between predictions and observations.
    """

    defines_error: list[PredictionError] = relation(owns=True)
    """
    The prediction errors a mismatch can be classified as.
    """

    defines_correction: list[CorrectiveAction] = relation(owns=True)
    """
    The corrective responses to prediction errors.
    """

    defines_model_update: list[ModelUpdate] = relation(owns=True)
    """
    The revisions of represented estimates driven by prediction errors.
    """

    defines_initial_state_estimate: list[InitialStateEstimate] = relation(owns=True)
    """
    The next-phase states supplied by successful comparisons.
    """

    defines_control_point: list[SensorimotorControlPoint] = relation(owns=True)
    """
    The sensorimotor checkpoints at physical events.
    """

    defines_control_point_exemption: list[BoundaryControlExemption] = relation(
        owns=True
    )
    """
    The boundaries that are justified in having no sensorimotor checkpoint.
    """
