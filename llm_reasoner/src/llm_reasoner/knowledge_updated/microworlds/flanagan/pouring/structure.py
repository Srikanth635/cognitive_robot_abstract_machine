"""
A prehension phase structure that transfers a substance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import POURING
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.value_partitions import (
    MaterialRegime,
    TransferTerminationReason,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.structure import (
    PrehensionPhaseStructure,
)
from llm_reasoner.knowledge_updated.ontological.properties import attribute
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@POURING.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class PouringPhaseStructure(PrehensionPhaseStructure):
    """
    A substance-transfer phase structure composed with prehension.

    Its continuous control is declared with the continuous predictions of the core.
    """

    material_regime: Optional[MaterialRegime] = attribute(
        functional=True, minimum_count=1
    )
    """
    The flow behavior governing the transferred substance.
    """

    termination_reason: Optional[TransferTerminationReason] = attribute(
        functional=True, minimum_count=1
    )
    """
    The condition that ends transfer in this task variant.
    """
