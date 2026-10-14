"""
The participants of a prehensile manipulation besides the actor and the manipulandum.

The effector makes the grasp, the grasp site is where it touches, and the supports bear
the object before liftoff and after touchdown. Like every role they are borne by actual
entities only once a slot is bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    SOMA,
    related_to,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    Manipulandum,
    ManipulationRole,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import PREHENSION
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    EvidenceStatus,
    Rigidity,
)


@PREHENSION.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(related_to(SOMA, "PrehensileEffector"),),
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class GraspingEffector(ManipulationRole):
    """
    The hand, gripper, or other effector establishing a grasp.
    """


@PREHENSION.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class GraspSite(ManipulationRole):
    """
    The manipulandum region at which the effector makes contact.
    """


@PREHENSION.concept(
    rigidity=Rigidity.ANTI_RIGID, level=ConceptLevel.OCCURRENCE, role_taker=Symbol
)
@dataclass(eq=False)
class PrehensileManipulandum(Manipulandum):
    """
    The object coupled to an effector by a prehensile grasp.
    """


@PREHENSION.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(related_to(SOMA, "Supporter"),),
    may_overlap_with=["FinalSupport"],
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class InitialSupport(ManipulationRole):
    """
    The surface bearing the manipulandum before liftoff.
    """


@PREHENSION.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(related_to(SOMA, "Supporter"),),
    may_overlap_with=["InitialSupport"],
    status=EvidenceStatus.EMPIRICAL_FINDING,
)
@dataclass(eq=False)
class FinalSupport(ManipulationRole):
    """
    The surface bearing the manipulandum after touchdown, which may be the surface that
    bore it before liftoff.
    """
