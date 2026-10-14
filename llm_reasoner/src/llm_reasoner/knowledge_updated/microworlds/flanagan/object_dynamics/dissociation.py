"""
A sensorimotor estimate and a perceptual judgment that disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import Estimate
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    DynamicsModel,
    PerceptualRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import OBJECT_DYNAMICS
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@OBJECT_DYNAMICS.concept(
    alignments=(subclass_of(DUL, "Situation"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-031",),
)
@dataclass(eq=False)
class RepresentationDissociation(KnowledgeEntity):
    """
    A sensorimotor estimate that stays accurate while a perceptual judgment stays
    biased.
    """

    sensorimotor_representation: Optional[DynamicsModel] = relation(
        functional=True, minimum_count=1
    )
    """
    The representation used for motor prediction and scaling.
    """

    perceptual_representation: Optional[PerceptualRepresentation] = relation(
        functional=True, minimum_count=1
    )
    """
    The representation used for conscious judgment.
    """

    sensorimotor_estimate: Optional[Estimate] = relation(
        functional=True, minimum_count=1
    )
    """
    The accurate estimate held by the sensorimotor representation.
    """

    perceptual_estimate: Optional[Estimate] = relation(functional=True, minimum_count=1)
    """
    The biased estimate held by the perceptual representation.
    """
