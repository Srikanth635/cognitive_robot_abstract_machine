"""
The events that bound a substance transfer.

A flow event changes the motion of the substance, and may also cross a threshold. Events
that cross a threshold or attain a setpoint name the kind of threshold or setpoint, so a
hypothesis that names another is refused without a separate rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    MechanicalEventType,
    SetpointAttained,
    ThresholdCrossingEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import (
    CORE,
    POURING,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.parameters import (
    FlowOnsetThreshold,
    LipAlignmentSetpoint,
    SpillThreshold,
    TransferredQuantitySetpoint,
    UprightAngleSetpoint,
)
from llm_reasoner.knowledge_updated.ontological.properties import relation
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

ENGINEERING = EvidenceStatus.ENGINEERING_EXTENSION

# %% flow


@POURING.concept(
    disjoint_siblings=True,
    disjoint_with=["ContactEventType"],
    may_overlap_with=["ThresholdCrossingEventType"],
    status=ENGINEERING,
)
@dataclass(eq=False)
class FlowEventType(MechanicalEventType):
    """
    An event constituted by a change in substance motion.

    Flow identity is orthogonal to threshold crossing, so a flow event may also be a
    threshold crossing.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class FlowOnset(FlowEventType, ThresholdCrossingEventType):
    """
    Substance first crosses the lip after the onset threshold is crossed.
    """

    crosses_threshold: Optional[FlowOnsetThreshold] = relation(
        functional=True,
        minimum_count=1,
        iri=CORE.namespace.property_term("crosses_threshold"),
    )
    """
    The tilt angle at which the substance begins crossing the lip.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class FlowCessation(FlowEventType):
    """
    Substance stops crossing the pouring lip.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class SpillEvent(FlowEventType, ThresholdCrossingEventType):
    """
    Substance flow exceeds the containment threshold outside the receptacle.
    """

    crosses_threshold: Optional[SpillThreshold] = relation(
        functional=True,
        minimum_count=1,
        iri=CORE.namespace.property_term("crosses_threshold"),
    )
    """
    The flow rate beyond which containment is violated.
    """


# %% control boundaries


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class TargetQuantityReached(SetpointAttained):
    """
    Transferred mass or volume meets the requested task quantity.
    """

    attains_setpoint: list[TransferredQuantitySetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The requested quantities whose attainment constitutes this event.
    """


@POURING.concept()
@dataclass(eq=False)
class SourceEmpty(SetpointAttained):
    """
    The remaining transferable quantity in the source reaches zero.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class LipAligned(SetpointAttained):
    """
    Commanded pose placing the lip over the receptacle opening has been reached.
    """

    attains_setpoint: list[LipAlignmentSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The lip alignments whose attainment constitutes this event.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class UprightRestored(SetpointAttained):
    """
    The container has been returned to a stable orientation.
    """

    attains_setpoint: list[UprightAngleSetpoint] = relation(
        minimum_count=1, iri=CORE.namespace.property_term("attains_setpoint")
    )
    """
    The upright orientations whose attainment constitutes this event.
    """
