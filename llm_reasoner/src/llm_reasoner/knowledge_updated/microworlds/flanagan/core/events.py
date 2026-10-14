"""
The events that end a phase.

A phase ends at a boundary event of one of three exclusive kinds: a physical transition the
controller can predict and detect, the attainment of a commanded setpoint, or an external
request. Physical events combine characteristics, so a lift-off both breaks contact and
crosses a force threshold. That is why the characteristics are classes to inherit from
together and not values of one property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.alignments import (
    DUL,
    subclass_of,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.parameters import (
    MechanicalThreshold,
    Setpoint,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ContactTransition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import CORE
from llm_reasoner.knowledge_updated.ontological.entities import KnowledgeEntity
from llm_reasoner.knowledge_updated.ontological.properties import (
    attribute,
    reference,
    relation,
)
from llm_reasoner.knowledge_updated.ontological.references import ConceptReference
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% every boundary event


@CORE.concept(
    alignments=(subclass_of(DUL, "EventType"),),
    abstract=True,
    disjoint_siblings=True,
)
@dataclass(eq=False)
class PhaseBoundaryEventType(KnowledgeEntity):
    """
    A discrete transition that can end a phase.
    """

    is_failure_event: bool = attribute(functional=True, default=False)
    """
    Whether the event marks failure and not intended completion.
    """


# %% physical events


@CORE.concept(status=EvidenceStatus.AUTHOR_HYPOTHESIS, claims=("FJ-004",))
@dataclass(eq=False)
class MechanicalEventType(PhaseBoundaryEventType):
    """
    A discrete physical transition the controller can predict and detect.
    """

    event_on_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The participants between which the event occurs.
    """


@CORE.concept(
    disjoint_siblings=True,
    may_overlap_with=["ThresholdCrossingEventType"],
    status=EvidenceStatus.AUTHOR_HYPOTHESIS,
    claims=("FJ-004",),
)
@dataclass(eq=False)
class ContactEventType(MechanicalEventType):
    """
    An event that changes whether its participants are in contact.
    """

    contact_transition: Optional[ContactTransition] = attribute(
        functional=True, minimum_count=1
    )
    """
    Whether the event establishes or breaks contact.
    """


@CORE.concept(disjoint_siblings=True)
@dataclass(eq=False)
class ThresholdCrossingEventType(MechanicalEventType):
    """
    An event that occurs when a mechanical quantity crosses a threshold.
    """

    crosses_threshold: Optional[MechanicalThreshold] = relation(
        functional=True, minimum_count=1
    )
    """
    The threshold crossed at this event.
    """


@CORE.concept(disjoint_siblings=True, status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class ForceDynamicEventType(ThresholdCrossingEventType):
    """
    A threshold crossing driven by contact or inertial force dynamics.
    """


# %% setpoint attainment


@CORE.concept(disjoint_siblings=True, status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class SetpointAttained(PhaseBoundaryEventType):
    """
    A commanded pose, force or task quantity has been reached.

    Reaching a setpoint is a control-level event and not a physical transition, so no
    sensory prediction is anchored to it.
    """

    event_on_role: list[RoleSlot] = relation(minimum_count=1)
    """
    The participants whose state reaches the setpoint.
    """

    attains_setpoint: list[Setpoint] = relation(minimum_count=1)
    """
    The setpoints whose attainment constitutes this event.
    """


# %% external requests


@CORE.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class ExternalTerminationEventType(PhaseBoundaryEventType):
    """
    A phase boundary imposed by a source outside the manipulation.
    """

    triggered_by: Optional[ConceptReference] = reference(
        functional=True, minimum_count=1
    )
    """
    The external request, policy or safety condition ending the phase.
    """
