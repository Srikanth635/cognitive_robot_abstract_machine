"""
A phase structure that also declares how grip force is controlled.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.structure import (
    ObjectDynamicsPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import PREHENSION
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.control import (
    AnticipatoryGripAdjustment,
    GripLoadCoupling,
)
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@PREHENSION.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class PrehensionPhaseStructure(ObjectDynamicsPhaseStructure):
    """
    A phase structure with explicit prehensile force-control relations.

    Each of its phases must be of a specific prehension kind.
    """

    defines_force_coupling: list[GripLoadCoupling] = relation(owns=True)
    """
    The functional grip-load relations declared by the structure.
    """

    defines_anticipatory_grip_adjustment: list[AnticipatoryGripAdjustment] = relation(
        owns=True
    )
    """
    The prediction-triggered grip adjustments declared by the structure.
    """
