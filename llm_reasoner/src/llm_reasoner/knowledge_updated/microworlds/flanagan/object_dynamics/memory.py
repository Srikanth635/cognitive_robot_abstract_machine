"""
What is retained between actions, and how retention changes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    related_to,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.control import (
    ModelUpdate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.representations import (
    ContextualRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    MemoryPersistence,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import OBJECT_DYNAMICS
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% memory


@OBJECT_DYNAMICS.concept(
    alignments=(related_to(DUL, "Description"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-009", "FJ-028"),
)
@dataclass(eq=False)
class SensorimotorMemory(KnowledgeEntity):
    """
    Memory that recruits one or more sensorimotor representations.
    """

    stores_representation: list[ContextualRepresentation] = relation(minimum_count=1)
    """
    The representations retained by the memory.
    """

    memory_persistence: Optional[MemoryPersistence] = attribute(
        functional=True, minimum_count=1
    )
    """
    Whether recruitment ends with object release.
    """


# %% updating


@OBJECT_DYNAMICS.concept(
    status=EvidenceStatus.EMPIRICAL_FINDING, claims=("FJ-010", "FJ-011")
)
@dataclass(eq=False)
class RetainedModelUpdate(ModelUpdate):
    """
    A revision of represented estimates that a memory retains.
    """

    persists_in: Optional[SensorimotorMemory] = relation(
        functional=True, minimum_count=1
    )
    """
    The memory in which the revised representation is retained.
    """
