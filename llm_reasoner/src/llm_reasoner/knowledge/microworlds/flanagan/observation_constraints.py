"""
Acceptance constraints owned by the action-observation module.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import ScientificStatus
from llm_reasoner.knowledge.microworlds.flanagan.observation import (
    ActorObserverPredictionMatch,
    ObservedActionPrediction,
)
from llm_reasoner.knowledge.ontological.constraints import Forbidden

# %% observation predicates


def observation_uses_same_actor_and_observer(
    prediction: ObservedActionPrediction,
) -> bool:
    """
    Return whether one role slot is used as both actor and observer.
    """
    return prediction.actor is not None and prediction.actor is prediction.observer


# %% observation constraints


OBSERVATION_CONSTRAINTS = [
    Forbidden(
        id="OBS-C60",
        applies_to=ActorObserverPredictionMatch,
        when=lambda matching: matching.scientific_status
        is not ScientificStatus.AUTHOR_HYPOTHESIS,
        message="actor-observer matching is not marked as an author hypothesis",
        rationale="Common predictions and covert plan matching are proposed rather "
        "than established by the reviewed evidence.",
    ),
    Forbidden(
        id="OBS-C61",
        applies_to=ObservedActionPrediction,
        when=observation_uses_same_actor_and_observer,
        message="actor and observer use the same role slot",
        rationale="Execution and observation are distinct participant roles in the "
        "action-observation experiments.",
    ),
]


__all__ = ["OBSERVATION_CONSTRAINTS"]
