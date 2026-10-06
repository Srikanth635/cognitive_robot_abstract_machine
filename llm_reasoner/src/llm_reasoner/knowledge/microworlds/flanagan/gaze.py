"""
Gaze-control vocabulary for predictive manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ActionPhase,
    MagnitudeRange,
    MechanicalEventType,
    ScientificAssertion,
    SensorimotorControlPoint,
    SensoryObservationType,
    Subgoal,
)
from llm_reasoner.knowledge.microworlds.flanagan.namespaces import FlanaganNamespace
from llm_reasoner.knowledge.ontological.decorators import (
    ConceptReference,
    KnowledgeEntity,
    ontological,
    prop,
    value_partition,
)

GAZE = FlanaganNamespace.GAZE
"""
Namespace for gaze-control concepts.
"""

# %% gaze mode


@value_partition(GAZE + "GazeMode")
class GazeMode(StrEnum):
    """
    Whether gaze anticipates or follows task-relevant motion.
    """

    PROACTIVE = "proactive"
    """
    Gaze reaches a forthcoming contact location before the hand or object.
    """

    REACTIVE = "reactive"
    """
    Gaze follows visible motion or responds after task evidence appears.
    """


# %% gaze targets and episodes


@ontological(iri=GAZE + "GazeTarget")
@dataclass(eq=False)
class GazeTarget(KnowledgeEntity):
    """
    A forthcoming contact location selected as a gaze target.
    """

    contact_location: ConceptReference | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "hasContactLocation",
    )
    """
    The task-relevant spatial location selected for fixation.
    """

    target_for_phase: list[ActionPhase] = prop(
        min_count=1,
        default_factory=list,
        iri=GAZE + "targetForPhase",
    )
    """
    The phases whose forthcoming contact is anticipated at this target.
    """


@ontological(iri=GAZE + "GazeEpisode")
@dataclass(eq=False)
class GazeEpisode(ScientificAssertion):
    """
    A fixation policy associated with one manipulation subgoal.
    """

    mode: GazeMode | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "hasGazeMode",
    )
    """
    Whether the gaze behavior is proactive or reactive.
    """

    target: GazeTarget | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "hasGazeTarget",
    )
    """
    The contact location selected for the episode.
    """

    lead_time: MagnitudeRange | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "hasGazeLeadTime",
    )
    """
    How long gaze precedes hand or held-object arrival at the target.
    """

    fixation_until: Subgoal | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "fixatesUntil",
    )
    """
    The subgoal whose completion releases fixation from the target.
    """

    monitors_contact: list[MechanicalEventType] = prop(
        min_count=1,
        default_factory=list,
        iri=GAZE + "monitorsContact",
    )
    """
    The contact events visually monitored during fixation.
    """


@ontological(iri=GAZE + "MultimodalAlignment")
@dataclass(eq=False)
class MultimodalAlignment(ScientificAssertion):
    """
    Alignment of visual contact evidence with other sensory evidence.
    """

    control_point: SensorimotorControlPoint | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "atControlPoint",
    )
    """
    The control point at which the observations are aligned.
    """

    visual_observation: SensoryObservationType | None = prop(
        functional=True,
        min_count=1,
        iri=GAZE + "hasVisualObservation",
    )
    """
    The visual evidence serving as the temporal reference.
    """

    aligned_observations: list[SensoryObservationType] = prop(
        min_count=1,
        default_factory=list,
        iri=GAZE + "alignsObservation",
    )
    """
    The tactile, proprioceptive, or auditory evidence aligned with vision.
    """


# %% gaze profile


@ontological(iri=GAZE + "GazeControlProfile")
@dataclass(eq=False)
class GazeControlProfile(KnowledgeEntity):
    """
    The gaze episodes and multimodal alignments for an action.
    """

    episodes: list[GazeEpisode] = prop(
        min_count=1,
        default_factory=list,
        iri=GAZE + "hasGazeEpisode",
    )
    """
    The gaze policies associated with successive manipulation subgoals.
    """

    alignments: list[MultimodalAlignment] = prop(
        default_factory=list,
        iri=GAZE + "hasMultimodalAlignment",
    )
    """
    The proposed cross-modal alignments at sensorimotor control points.
    """
