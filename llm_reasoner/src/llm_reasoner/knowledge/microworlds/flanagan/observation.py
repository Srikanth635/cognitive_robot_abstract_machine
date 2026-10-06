"""
Action-observation vocabulary kept separate from action execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    Estimate,
    ManipulationRole,
    RoleSlot,
    ScientificAssertion,
    SensoryEventPrediction,
    SensorySignalType,
)
from llm_reasoner.knowledge.microworlds.flanagan.gaze import GazeEpisode, GazeTarget
from llm_reasoner.knowledge.microworlds.flanagan.namespaces import FlanaganNamespace
from llm_reasoner.knowledge.ontological.decorators import (
    KnowledgeEntity,
    ontological,
    prop,
    value_partition,
)
from llm_reasoner.knowledge.ontological.rigidity import ConceptLevel, Rigidity

OBSERVATION = FlanaganNamespace.OBSERVATION
"""
Namespace for action-observation concepts.
"""

SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"
"""
SOMA namespace used for external alignments.
"""

# %% observer conditions


@value_partition(OBSERVATION + "HandVisibility")
class HandVisibility(StrEnum):
    """
    How much of the actor's hand motion is available to an observer.
    """

    VISIBLE = "visible"
    """
    The task-relevant hand trajectory is visible.
    """

    PARTIALLY_OCCLUDED = "partially_occluded"
    """
    Only part of the task-relevant hand trajectory is visible.
    """

    HIDDEN = "hidden"
    """
    The task-relevant hand trajectory is unavailable to the observer.
    """


@ontological(
    iri=OBSERVATION + "Observer",
    aligns_to=SOMA + "AgentRole",
    rigidity=Rigidity.ANTI_RIGID,
    role_taker="Symbol",
    level=ConceptLevel.OCCURRENCE,
    disjoint_with=["Actor"],
)
@dataclass(eq=False)
class Observer(ManipulationRole):
    """
    The agent observing rather than executing the manipulation.
    """


# %% observed action prediction


@ontological(iri=OBSERVATION + "ObservedActionPrediction")
@dataclass(eq=False)
class ObservedActionPrediction(KnowledgeEntity):
    """
    An observer's prediction about an actor's forthcoming contact.
    """

    actor: RoleSlot | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "hasActor",
    )
    """
    The role slot for the agent performing the observed action.
    """

    observer: RoleSlot | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "hasObserver",
    )
    """
    The role slot for the agent observing the action.
    """

    observed_hand_visibility: HandVisibility | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "hasObservedHandVisibility",
    )
    """
    The availability of the actor's task-relevant hand motion.
    """

    predicted_contact_site: GazeTarget | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "predictsContactSite",
    )
    """
    The forthcoming actor contact location inferred by the observer.
    """

    available_modalities: list[SensorySignalType] = prop(
        min_count=1,
        default_factory=list,
        iri=OBSERVATION + "hasAvailableModality",
    )
    """
    The sensory modalities available to the observer.
    """

    inferred_hidden_properties: list[Estimate] = prop(
        default_factory=list,
        iri=OBSERVATION + "infersHiddenProperty",
    )
    """
    The hidden object properties inferred from the observed action.
    """

    observer_gaze: GazeEpisode | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "hasObserverGaze",
    )
    """
    The observer's proactive or reactive gaze behavior.
    """


@ontological(iri=OBSERVATION + "ActorObserverPredictionMatch")
@dataclass(eq=False)
class ActorObserverPredictionMatch(ScientificAssertion):
    """
    The proposed sharing of predictions between actor and observer.
    """

    actor: RoleSlot | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "hasActor",
    )
    """
    The acting participant whose prediction is proposed to be matched.
    """

    observer: RoleSlot | None = prop(
        functional=True,
        min_count=1,
        iri=OBSERVATION + "hasObserver",
    )
    """
    The observing participant proposed to match the prediction.
    """

    shared_predictions: list[SensoryEventPrediction] = prop(
        min_count=1,
        default_factory=list,
        iri=OBSERVATION + "sharesPrediction",
    )
    """
    The sensory predictions proposed to be common to both participants.
    """


# %% observation profile


@ontological(iri=OBSERVATION + "ActionObservationProfile")
@dataclass(eq=False)
class ActionObservationProfile(KnowledgeEntity):
    """
    The predictions and matching hypotheses for observed manipulation.
    """

    predictions: list[ObservedActionPrediction] = prop(
        min_count=1,
        default_factory=list,
        iri=OBSERVATION + "hasObservedActionPrediction",
    )
    """
    The observer predictions supported by the experimental conditions.
    """

    matching_hypotheses: list[ActorObserverPredictionMatch] = prop(
        default_factory=list,
        iri=OBSERVATION + "hasMatchingHypothesis",
    )
    """
    The proposed actor-observer prediction matches.
    """
