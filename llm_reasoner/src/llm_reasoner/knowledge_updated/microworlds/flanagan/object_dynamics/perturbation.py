"""
Changing the dynamics a model represents, and what carries over.

A load or a rotation is applied to a represented dynamics system. Two perturbations that
are matched physically, one on the object and one on the arm, let a transfer compare
adaptation across them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    DynamicsModel,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    CoordinateFrame,
    PerturbationKind,
    TransferOutcome,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import OBJECT_DYNAMICS
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@OBJECT_DYNAMICS.concept(
    alignments=(subclass_of(DUL, "Situation"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-026",),
)
@dataclass(eq=False)
class DynamicsPerturbation(KnowledgeEntity):
    """
    A load or rotation applied to one represented dynamics system.
    """

    perturbation_kind: Optional[PerturbationKind] = attribute(
        functional=True, minimum_count=1
    )
    """
    The physical transformation used to assess dynamics transfer.
    """

    affects_model: Optional[DynamicsModel] = relation(functional=True, minimum_count=1)
    """
    The object or limb dynamics subjected to the perturbation.
    """

    coordinate_frame: Optional[CoordinateFrame] = attribute(
        functional=True, minimum_count=1
    )
    """
    The frame in which the perturbation is expressed.
    """

    matched_with: list[DynamicsPerturbation] = relation()
    """
    The perturbations matched physically with this one for a transfer comparison.
    """


@OBJECT_DYNAMICS.concept(
    alignments=(subclass_of(DUL, "Situation"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-030",),
)
@dataclass(eq=False)
class DynamicsTransfer(KnowledgeEntity):
    """
    The adaptation result between two perturbation contexts.
    """

    from_perturbation: Optional[DynamicsPerturbation] = relation(
        functional=True, minimum_count=1
    )
    """
    The context in which adaptation was acquired.
    """

    to_perturbation: Optional[DynamicsPerturbation] = relation(
        functional=True, minimum_count=1
    )
    """
    The context in which transfer was assessed.
    """

    transfer_outcome: Optional[TransferOutcome] = attribute(
        functional=True, minimum_count=1
    )
    """
    Whether adaptation generalized across the contexts.
    """
