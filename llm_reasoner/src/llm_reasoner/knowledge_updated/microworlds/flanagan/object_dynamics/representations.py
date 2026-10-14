"""
Represented estimates with the frame and purpose they are recruited for.

A representation holds estimates. Here each also says in which frame its content is
encoded, for which function it is recruited, and about which participants. Content used to
predict and scale motor commands is kept apart from content used for perceptual judgment,
because the two can disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.models import (
    ObjectRepresentation,
    EstimateRepresentation,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    DynamicsEstimate,
    Estimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ParameterValueForm,
    PhysicalDimension,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.commands import (
    EfferenceCopy,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.value_partitions import (
    CoordinateFrame,
    RepresentationPurpose,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    OBJECT_DYNAMICS,
)
from llm_reasoner.knowledge_updated.ontological.properties import attribute, relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% every representation


@OBJECT_DYNAMICS.concept(
    abstract=True,
    disjoint_siblings=True,
    may_overlap_with=["ObjectRepresentation"],
    status=EvidenceStatus.ENGINEERING_EXTENSION,
    claims=("FJ-029", "FJ-031"),
)
@dataclass(eq=False)
class ContextualRepresentation(EstimateRepresentation):
    """
    Represented estimates with an explicit frame, purpose and set of participants.
    """

    holds_estimate: list[Estimate] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("holds_estimate")
    )
    """
    The estimates stored in the representation.
    """

    coordinate_frame: Optional[CoordinateFrame] = attribute(
        functional=True, minimum_count=1
    )
    """
    The frame in which the represented content is encoded.
    """

    purpose: Optional[RepresentationPurpose] = attribute(
        functional=True, minimum_count=1
    )
    """
    The function for which the represented content is recruited.
    """

    represents_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The participant roles whose properties or dynamics are represented.
    """


# %% content for control


@OBJECT_DYNAMICS.concept(
    abstract=True,
    disjoint_siblings=True,
    fixed_values={"purpose": RepresentationPurpose.SENSORIMOTOR_CONTROL},
    status=EvidenceStatus.SUPPORTED_INTERPRETATION,
    claims=("FJ-027",),
)
@dataclass(eq=False)
class DynamicsModel(ContextualRepresentation):
    """
    A represented mapping from command-related input to mechanical change.
    """

    uses_efference_copy: list[EfferenceCopy] = relation(minimum_count=1)
    """
    The outgoing command copies used to predict the load the commands will meet.
    """


@OBJECT_DYNAMICS.concept(
    status=EvidenceStatus.SUPPORTED_INTERPRETATION, claims=("FJ-008", "FJ-027")
)
@dataclass(eq=False)
class ObjectDynamicsModel(DynamicsModel, ObjectRepresentation):
    """
    A dynamics model recruited for a manipulated object.
    """


@OBJECT_DYNAMICS.concept(
    status=EvidenceStatus.SUPPORTED_INTERPRETATION, claims=("FJ-027",)
)
@dataclass(eq=False)
class LimbDynamicsModel(DynamicsModel):
    """
    A dynamics model representing the actor's limb or effector.
    """


# %% content for judgment


@OBJECT_DYNAMICS.concept(
    fixed_values={"purpose": RepresentationPurpose.PERCEPTUAL_JUDGMENT},
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-031",),
)
@dataclass(eq=False)
class PerceptualRepresentation(ContextualRepresentation):
    """
    Represented content used for perceptual property judgments.
    """


# %% estimates of load


@OBJECT_DYNAMICS.concept(
    fixed_values={
        "value_form": ParameterValueForm.QUANTITATIVE,
        "dimension": PhysicalDimension.FORCE,
    },
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-025",),
)
@dataclass(eq=False)
class LoadForceEstimate(DynamicsEstimate):
    """
    An estimate of force arising from object or limb dynamics.
    """
