"""
The physical and control events that bound a prehensile manipulation.

Contact events say whether they establish or break contact, force-dynamic events name the
threshold they cross, and setpoint events name the kind of setpoint they attain. Those
statements are part of each concept, so a hypothesis that contradicts one is refused
without a separate rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ContactEventType,
    ForceDynamicEventType,
    SetpointAttained,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.roles import RoleSlot
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    ContactTransition,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    PREHENSION,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.parameters import (
    LiftClearanceSetpoint,
    LiftoffThreshold,
    ReachPoseSetpoint,
    SlipThreshold,
    TouchdownThreshold,
    TransportPoseSetpoint,
    WithdrawalDistanceSetpoint,
)
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

# %% reaching and contact


@PREHENSION.concept()
@dataclass(eq=False)
class ReachTargetReached(SetpointAttained):
    """
    The effector reaches the commanded pre-contact pose.
    """

    attains_setpoint: list[ReachPoseSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The pre-contact pose setpoints whose attainment constitutes this event.
    """


@PREHENSION.concept(
    fixed_values={"contact_transition": ContactTransition.ESTABLISHES},
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-004",),
)
@dataclass(eq=False)
class ContactEstablishment(ContactEventType):
    """
    The effector first contacts the manipulandum.
    """


@PREHENSION.concept(
    fixed_values={"contact_transition": ContactTransition.BREAKS},
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-004",),
)
@dataclass(eq=False)
class ContactBreak(ContactEventType):
    """
    The effector loses contact with the manipulandum.
    """


# %% lifting and lowering


@PREHENSION.concept(
    fixed_values={"contact_transition": ContactTransition.BREAKS},
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-004", "FJ-010"),
)
@dataclass(eq=False)
class Liftoff(ContactEventType, ForceDynamicEventType):
    """
    Load exceeds weight and the manipulandum separates from its support.
    """

    crosses_threshold: Optional[LiftoffThreshold] = relation(
        functional=True,
        minimum_count=1,
        iri=CORE.namespace.property_term("crosses_threshold"),
    )
    """
    The load-force threshold that corresponds to the weight of the manipulandum.
    """


@PREHENSION.concept(
    fixed_values={"contact_transition": ContactTransition.ESTABLISHES},
    status=EvidenceStatus.EMPIRICAL_FINDING,
    claims=("FJ-004",),
)
@dataclass(eq=False)
class Touchdown(ContactEventType, ForceDynamicEventType):
    """
    The manipulandum contacts its destination support.
    """

    crosses_threshold: Optional[TouchdownThreshold] = relation(
        functional=True,
        minimum_count=1,
        iri=CORE.namespace.property_term("crosses_threshold"),
    )
    """
    The support-reaction threshold that registers the touchdown.
    """


@PREHENSION.concept(status=EvidenceStatus.EMPIRICAL_FINDING)
@dataclass(eq=False)
class SlipEvent(ForceDynamicEventType):
    """
    Tangential load exceeds the frictional capacity of the grasp.
    """

    crosses_threshold: Optional[SlipThreshold] = relation(
        functional=True,
        minimum_count=1,
        iri=CORE.namespace.property_term("crosses_threshold"),
    )
    """
    The minimum grip force that the tangential load has come to exceed.
    """


# %% control boundaries


@PREHENSION.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class LiftClearanceReached(SetpointAttained):
    """
    The manipulandum reaches its commanded support clearance.
    """

    attains_setpoint: list[LiftClearanceSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The clearance setpoints whose attainment constitutes this event.
    """


@PREHENSION.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class StableHoldReached(SetpointAttained):
    """
    The requested stable holding state has been reached.
    """


@PREHENSION.concept()
@dataclass(eq=False)
class TransportTargetReached(SetpointAttained):
    """
    The held manipulandum reaches its commanded transport destination.
    """

    attains_setpoint: list[TransportPoseSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The transport pose setpoints whose attainment constitutes this event.
    """


@PREHENSION.concept()
@dataclass(eq=False)
class ForcesUnloaded(SetpointAttained):
    """
    Support transfer completes and grip and load forces reach their low targets.
    """

    weight_supported_by: list[RoleSlot] = relation(minimum_count=1)
    """
    The support roles bearing manipulandum weight at force completion.
    """


@PREHENSION.concept(status=EvidenceStatus.ENGINEERING_EXTENSION)
@dataclass(eq=False)
class WithdrawalComplete(SetpointAttained):
    """
    The effector reaches its post-release withdrawal target.
    """

    attains_setpoint: list[WithdrawalDistanceSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The withdrawal distance setpoints whose attainment constitutes this event.
    """
