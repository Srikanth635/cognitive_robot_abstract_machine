"""
Acceptance constraints owned by the gaze-control module.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    PhysicalDimension,
    ScientificStatus,
    VisualSignal,
)
from llm_reasoner.knowledge.microworlds.flanagan.gaze import (
    GazeEpisode,
    MultimodalAlignment,
)
from llm_reasoner.knowledge.ontological.constraints import Forbidden

# %% gaze predicates


def gaze_episode_has_invalid_lead_dimension(episode: GazeEpisode) -> bool:
    """
    Return whether gaze lead is expressed in a non-temporal dimension.
    """
    if episode.lead_time is None:
        return False
    return episode.lead_time.physical_dimension is not PhysicalDimension.TIME


def alignment_has_invalid_modalities(alignment: MultimodalAlignment) -> bool:
    """
    Return whether visual evidence is not aligned with non-visual evidence.
    """
    if alignment.visual_observation is None:
        return False
    if not isinstance(alignment.visual_observation.modality, VisualSignal):
        return True
    return any(
        isinstance(observation.modality, VisualSignal)
        for observation in alignment.aligned_observations
    )


# %% gaze constraints


GAZE_CONSTRAINTS = [
    Forbidden(
        id="GAZE-C59a",
        applies_to=GazeEpisode,
        when=lambda episode: episode.scientific_status
        is not ScientificStatus.EMPIRICAL_FINDING,
        message="proactive and reactive gaze behavior is not marked empirical",
        rationale="The reviewed gaze timing and fixation behavior is an empirical "
        "finding.",
    ),
    Forbidden(
        id="GAZE-C59b",
        applies_to=MultimodalAlignment,
        when=lambda alignment: alignment.scientific_status
        is not ScientificStatus.AUTHOR_HYPOTHESIS,
        message="multimodal alignment is not marked as an author hypothesis",
        rationale="The proposed monitoring and alignment function is not established "
        "as a shared mechanism.",
    ),
    Forbidden(
        id="GAZE-C59c",
        applies_to=GazeEpisode,
        when=gaze_episode_has_invalid_lead_dimension,
        message="gaze lead is not expressed as time",
        rationale="Gaze lead denotes temporal precedence over hand or object arrival.",
    ),
    Forbidden(
        id="GAZE-C59d",
        applies_to=MultimodalAlignment,
        when=alignment_has_invalid_modalities,
        message="multimodal alignment does not align visual with non-visual evidence",
        rationale="The proposed alignment relates visual contact evidence to tactile, "
        "proprioceptive, or auditory evidence.",
    ),
]


__all__ = ["GAZE_CONSTRAINTS"]
