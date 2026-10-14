"""
The controller's own representation of body and environment.

A forward model predicts the sensory consequences of a command and an inverse model finds
the command that reaches a desired state. A representation is different in kind: it is
the content, the estimates, that such models use. The two are not presumed disjoint, since
one object can both hold content and perform a function.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import Estimate
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% models that predict or invert


@CORE.concept(
    alignments=(subclass_of(DUL, "Description"),),
    abstract=True,
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
)
@dataclass(eq=False)
class InternalModel(KnowledgeEntity):
    """
    The controller's own representation of body and environment, used to predict and to
    invert.
    """


@CORE.concept(
    disjoint_with=["InverseModel"],
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-002",),
)
@dataclass(eq=False)
class ForwardModel(InternalModel):
    """
    Predicts sensory consequences from outgoing motor commands.

    Supplies the predictions that are compared at control points.
    """


@CORE.concept(status=EvidenceStatus.SUPPORTED_INTERPRETATION)
@dataclass(eq=False)
class InverseModel(InternalModel):
    """
    Determines the motor command required to reach a desired state.
    """


# %% represented content


@CORE.concept(abstract=True, may_overlap_with=["ForwardModel", "InverseModel"])
@dataclass(eq=False)
class EstimateRepresentation(InternalModel):
    """
    Represented content that explicitly owns one or more estimates.
    """

    holds_estimate: list[Estimate] = relation()
    """
    The estimates stored as content of the representation.
    """


@CORE.concept(status=EvidenceStatus.SUPPORTED_INTERPRETATION)
@dataclass(eq=False)
class ObjectRepresentation(EstimateRepresentation):
    """
    Stored beliefs about an object's physical properties, accumulated over prior
    manipulations.
    """
