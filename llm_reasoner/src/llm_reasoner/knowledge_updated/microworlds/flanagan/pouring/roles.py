"""
The participants of a substance transfer.

The container that is grasped and tilted is the manipulandum, and the substance that is
poured is a different participant: the task is about the substance, and what is
manipulated is the container.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    SOMA,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import (
    ManipulationRole,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import POURING
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.roles import (
    PrehensileManipulandum,
)
from llm_reasoner.knowledge_updated.ontological.vocabulary import (
    ConceptLevel,
    EvidenceStatus,
    Rigidity,
)


@POURING.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(subclass_of(SOMA, "Container"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class SourceContainer(PrehensileManipulandum):
    """
    The vessel that is grasped and tilted.

    Unlike a rigid manipulandum its inertial properties change during the action, which
    is the distinctive demand pouring places on the framework.
    """


@POURING.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(subclass_of(SOMA, "Destination"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class TargetReceptacle(ManipulationRole):
    """
    The vessel or surface receiving the substance.
    """


@POURING.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    alignments=(subclass_of(SOMA, "PouredObject"),),
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class PouredSubstance(ManipulationRole):
    """
    The substance transferred, as distinct from the container that is manipulated.
    """


@POURING.concept(
    rigidity=Rigidity.ANTI_RIGID,
    level=ConceptLevel.OCCURRENCE,
    role_taker=Symbol,
    status=EvidenceStatus.ENGINEERING_EXTENSION,
)
@dataclass(eq=False)
class PouringLip(ManipulationRole):
    """
    The edge over which the substance leaves the source container.

    The geometric locus of the flow-onset event.
    """
