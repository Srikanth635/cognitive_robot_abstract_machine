"""
What a controller expects to sense, and what it can be compared with.

A prediction is held before evidence is registered, and an observation type describes the
evidence that can arrive. They are kept separate so that they can agree or disagree.
Both are anchored to physical events: reaching a setpoint is a control-level event, so no
sensory evidence is attached to it. The channel is a value, the sensory modality, and not
a kind of class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import ForwardModel
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import Setpoint
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.quantities import (
    MagnitudeRange,
    TemporalWindow,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    SensoryModality,
    TrajectoryTrend,
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

# %% predictions and observations of events


@CORE.concept(
    alignments=(subclass_of(DUL, "Description"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-002",),
)
@dataclass(eq=False)
class SensoryEventPrediction(KnowledgeEntity):
    """
    A model-generated sensory consequence expected at a mechanical event.
    """

    modality: Optional[SensoryModality] = attribute(functional=True, minimum_count=1)
    """
    The sensory channel in which the consequence is expected.
    """

    predicts_event: Optional[MechanicalEventType] = relation(
        functional=True, exclusive_group="predicted_content"
    )
    """
    The expected event. Exactly one of the expected event and the expected feature is
    given.
    """

    predicts_feature: Optional[ConceptReference] = reference(
        functional=True, exclusive_group="predicted_content"
    )
    """
    The expected signal feature. Exactly one of the expected event and the expected
    feature is given.
    """

    caused_by_event: Optional[MechanicalEventType] = relation(
        functional=True, minimum_count=1
    )
    """
    The mechanical event expected to produce the sensory consequence.
    """

    expected_timing: Optional[TemporalWindow] = relation(
        functional=True, minimum_count=1
    )
    """
    The interval in which the consequence is expected.
    """

    expected_on_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The participant roles at which the consequence is expected.
    """

    expected_magnitude: Optional[MagnitudeRange] = relation(functional=True)
    """
    The expected magnitude, when the prediction is quantitative.
    """

    tolerance: Optional[MagnitudeRange] = relation(functional=True)
    """
    The admissible quantitative deviation from the expected magnitude.
    """

    generated_by: Optional[ForwardModel] = relation(functional=True, minimum_count=1)
    """
    The forward model that produced the prediction.
    """


@CORE.concept(
    alignments=(subclass_of(DUL, "EventType"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-012",),
)
@dataclass(eq=False)
class SensoryObservationType(KnowledgeEntity):
    """
    A type-level sensory result available for comparison with a prediction.
    """

    modality: Optional[SensoryModality] = attribute(functional=True, minimum_count=1)
    """
    The sensory channel carrying the observation.
    """

    observes_event: Optional[MechanicalEventType] = relation(
        functional=True, exclusive_group="observed_content"
    )
    """
    The observed event. Exactly one of the observed event and the observed feature is
    given.
    """

    observes_feature: Optional[ConceptReference] = reference(
        functional=True, exclusive_group="observed_content"
    )
    """
    The observed feature. Exactly one of the observed event and the observed feature is
    given.
    """

    observed_at: Optional[TemporalWindow] = relation(functional=True, minimum_count=1)
    """
    The interval in which the observation occurs.
    """

    observed_on_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The participant roles involved in the observation.
    """

    observed_magnitude: Optional[MagnitudeRange] = relation(functional=True)
    """
    The observed magnitude, when the observation is quantitative.
    """


# %% continuous prediction


@CORE.concept(alignments=(subclass_of(DUL, "Description"),))
@dataclass(eq=False)
class SignalTrajectory(KnowledgeEntity):
    """
    A structured change in one sensory feature over a time interval.
    """

    signal_feature: Optional[ConceptReference] = reference(
        functional=True, minimum_count=1
    )
    """
    The sensory feature whose evolution is predicted.
    """

    trend: Optional[TrajectoryTrend] = attribute(functional=True, minimum_count=1)
    """
    The expected direction of feature change.
    """

    expected_range: Optional[MagnitudeRange] = relation(
        functional=True, minimum_count=1
    )
    """
    The range traversed or maintained by the feature.
    """

    expected_timing: Optional[TemporalWindow] = relation(functional=True)
    """
    The interval over which the trajectory is expected.
    """


@CORE.concept(
    alignments=(subclass_of(DUL, "Description"),),
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
)
@dataclass(eq=False)
class ContinuousSignalPrediction(KnowledgeEntity):
    """
    A model-generated sensory trajectory that is compared throughout a phase.

    The phases in which it runs name it, not the other way round.
    """

    modality: Optional[SensoryModality] = attribute(functional=True, minimum_count=1)
    """
    The sensory channel carrying the predicted trajectory.
    """

    predicted_trajectory: Optional[SignalTrajectory] = relation(
        functional=True, minimum_count=1
    )
    """
    The structured feature change expected during the phase.
    """

    tolerance: Optional[MagnitudeRange] = relation(functional=True, minimum_count=1)
    """
    The admissible deviation from the predicted trajectory.
    """

    generated_by: Optional[ForwardModel] = relation(functional=True, minimum_count=1)
    """
    The forward model producing the prediction.
    """

    comparison_target: Optional[SensoryObservationType] = relation(
        functional=True, minimum_count=1
    )
    """
    The observation type continuously compared with the prediction.
    """

    controlled_variable: Optional[Setpoint] = relation(functional=True, minimum_count=1)
    """
    The setpoint adjusted from the comparison error.
    """
