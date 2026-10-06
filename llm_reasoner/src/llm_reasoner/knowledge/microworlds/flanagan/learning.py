"""
Visuomotor-learning vocabulary for changes across practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ScientificAssertion,
    SensoryObservationType,
)
from llm_reasoner.knowledge.microworlds.flanagan.gaze import GazeMode
from llm_reasoner.knowledge.microworlds.flanagan.namespaces import FlanaganNamespace
from llm_reasoner.knowledge.microworlds.flanagan.object_dynamics import CoordinateFrame
from llm_reasoner.knowledge.ontological.decorators import (
    ConceptReference,
    KnowledgeEntity,
    ontological,
    prop,
    value_partition,
)

LEARNING = FlanaganNamespace.LEARNING
"""
Namespace for visuomotor-learning concepts.
"""

# %% learning distinctions


@value_partition(LEARNING + "LearningStage")
class LearningStage(StrEnum):
    """
    The learner's stage of control across practice.
    """

    EXPLORATION = "exploration"
    """
    The learner searches for a usable action-to-outcome relation.
    """

    SKILL_ACQUISITION = "skill_acquisition"
    """
    The learner stabilizes a newly discovered sensorimotor mapping.
    """

    SKILL_REFINEMENT = "skill_refinement"
    """
    The learner improves an already functional mapping.
    """


@value_partition(LEARNING + "LearningProcessKind")
class LearningProcessKind(StrEnum):
    """
    Whether learning discovers a rule or adapts an existing mapping.
    """

    NOVEL_RULE_DISCOVERY = "novel_rule_discovery"
    """
    The learner identifies a previously unknown mapping rule.
    """

    EXISTING_MAPPING_ADAPTATION = "existing_mapping_adaptation"
    """
    The learner recalibrates a mapping whose rule is already known.
    """


# %% sensorimotor maps


@ontological(iri=LEARNING + "SensorimotorMap")
@dataclass(eq=False)
class SensorimotorMap(KnowledgeEntity):
    """
    A represented mapping from sensed state to controlled outcome.
    """

    input_frame: CoordinateFrame | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "hasInputFrame",
    )
    """
    The coordinate frame in which the controlled input is encoded.
    """

    output_frame: CoordinateFrame | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "hasOutputFrame",
    )
    """
    The coordinate frame in which the task outcome is encoded.
    """

    mapping_rule: ConceptReference | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "hasMappingRule",
    )
    """
    The rule connecting the encoded input and outcome.
    """


@ontological(iri=LEARNING + "SensorimotorMapUpdate")
@dataclass(eq=False)
class SensorimotorMapUpdate(KnowledgeEntity):
    """
    A practice-dependent update to a sensorimotor mapping.
    """

    updates_map: SensorimotorMap | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "updatesMap",
    )
    """
    The sensorimotor map changed by learning.
    """

    from_stage: LearningStage | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "fromLearningStage",
    )
    """
    The learning stage before the update.
    """

    to_stage: LearningStage | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "toLearningStage",
    )
    """
    The learning stage after the update.
    """

    based_on_observations: list[SensoryObservationType] = prop(
        min_count=1,
        default_factory=list,
        iri=LEARNING + "basedOnObservation",
    )
    """
    The task evidence supporting the mapping update.
    """


# %% learning episodes


@ontological(iri=LEARNING + "VisuomotorLearningEpisode")
@dataclass(eq=False)
class VisuomotorLearningEpisode(ScientificAssertion):
    """
    The gaze and mapping state associated with one learning stage.
    """

    stage: LearningStage | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "hasLearningStage",
    )
    """
    The learner's stage during the episode.
    """

    gaze_mode: GazeMode | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "hasGazeMode",
    )
    """
    Whether gaze follows motion or proactively targets the goal.
    """

    learning_process: LearningProcessKind | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "hasLearningProcess",
    )
    """
    Whether the episode discovers or adapts a mapping rule.
    """

    sensorimotor_map: SensorimotorMap | None = prop(
        functional=True,
        min_count=1,
        iri=LEARNING + "usesSensorimotorMap",
    )
    """
    The sensorimotor map being learned or refined.
    """


@ontological(iri=LEARNING + "VisuomotorLearningProfile")
@dataclass(eq=False)
class VisuomotorLearningProfile(KnowledgeEntity):
    """
    An ordered sequence of practice stages and map updates.
    """

    episodes: list[VisuomotorLearningEpisode] = prop(
        min_count=1,
        default_factory=list,
        iri=LEARNING + "hasLearningEpisode",
    )
    """
    The learning episodes ordered from exploration through refinement.
    """

    map_updates: list[SensorimotorMapUpdate] = prop(
        default_factory=list,
        iri=LEARNING + "hasMapUpdate",
    )
    """
    The sensorimotor-map updates occurring between learning stages.
    """
