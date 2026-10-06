"""
Acceptance constraints owned by the visuomotor-learning module.
"""

from __future__ import annotations

from itertools import pairwise

from llm_reasoner.knowledge.microworlds.flanagan.core import ScientificStatus
from llm_reasoner.knowledge.microworlds.flanagan.gaze import GazeMode
from llm_reasoner.knowledge.microworlds.flanagan.learning import (
    LearningStage,
    VisuomotorLearningEpisode,
    VisuomotorLearningProfile,
)
from llm_reasoner.knowledge.ontological.constraints import Forbidden

# %% learning predicates


def episode_has_incompatible_gaze_mode(episode: VisuomotorLearningEpisode) -> bool:
    """
    Return whether a terminal learning stage uses the opposite gaze mode.
    """
    if episode.stage is LearningStage.EXPLORATION:
        return episode.gaze_mode is not GazeMode.REACTIVE
    if episode.stage is LearningStage.SKILL_REFINEMENT:
        return episode.gaze_mode is not GazeMode.PROACTIVE
    return False


def profile_has_reversed_learning_stages(
    profile: VisuomotorLearningProfile,
) -> bool:
    """
    Return whether practice episodes move backward through learning stages.
    """
    ordered_stages = (
        LearningStage.EXPLORATION,
        LearningStage.SKILL_ACQUISITION,
        LearningStage.SKILL_REFINEMENT,
    )
    stage_positions = {stage: index for index, stage in enumerate(ordered_stages)}
    positions = [
        stage_positions[episode.stage]
        for episode in profile.episodes
        if episode.stage in stage_positions
    ]
    return any(
        current_position > next_position
        for current_position, next_position in pairwise(positions)
    )


# %% learning constraints


LEARNING_CONSTRAINTS = [
    Forbidden(
        id="LEARN-C62a",
        applies_to=VisuomotorLearningEpisode,
        when=episode_has_incompatible_gaze_mode,
        message="learning stage has an incompatible gaze mode",
        rationale="Exploration is reactive, while refined control proactively targets "
        "the goal.",
    ),
    Forbidden(
        id="LEARN-C62b",
        applies_to=VisuomotorLearningEpisode,
        when=lambda episode: episode.scientific_status
        is not ScientificStatus.EMPIRICAL_FINDING,
        message="stage-dependent gaze behavior is not marked empirical",
        rationale="The reactive-to-proactive change across practice is experimentally "
        "reported.",
    ),
    Forbidden(
        id="LEARN-C63",
        applies_to=VisuomotorLearningProfile,
        when=profile_has_reversed_learning_stages,
        message="learning episodes reverse the empirical stage progression",
        rationale="The represented progression runs from exploration through "
        "acquisition to refinement.",
    ),
]


__all__ = ["LEARNING_CONSTRAINTS"]
