"""
The states of the world that prehensile phases demand or bring about.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    SOMA,
    related_to,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.conditions import (
    PhaseConditionType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import PREHENSION
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus


@PREHENSION.concept(
    alignments=(related_to(SOMA, "ContactState"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class EffectorObjectContact(PhaseConditionType):
    """
    The effector is mechanically in contact with the manipulandum.
    """


@PREHENSION.concept(
    alignments=(related_to(SOMA, "ContactState"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class ObjectSupportContact(PhaseConditionType):
    """
    The manipulandum is mechanically in contact with a support.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class GraspStability(PhaseConditionType):
    """
    Grip force remains above the slip threshold by its safety margin.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class ObjectClearOfSupport(PhaseConditionType):
    """
    No support bears the manipulandum's weight.
    """


@PREHENSION.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class EffectorPositioned(PhaseConditionType):
    """
    The effector is positioned for contact establishment.
    """


@PREHENSION.concept()
@dataclass(eq=False)
class SafeContactForce(PhaseConditionType):
    """
    Contact force remains within the protective response limit.
    """
