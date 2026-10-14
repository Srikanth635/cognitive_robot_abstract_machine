"""
The phases of a substance transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.phases import ActionPhase
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.value_partitions import (
    CouplingRegime,
    RegulationMode,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import POURING
from llm_reasoner.knowledge_updated.ontological.vocabulary import EvidenceStatus

ENGINEERING = EvidenceStatus.ENGINEERING_EXTENSION


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class AlignmentPhase(ActionPhase):
    """
    Positioning of the lip over the receptacle prior to tilting.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class TiltToFlowPhase(ActionPhase):
    """
    Rotation of the source container until the substance reaches the lip and flow
    begins.

    Ends at flow onset.
    """


@POURING.concept(
    fixed_values={"coupling_regime": CouplingRegime.COUPLED},
    allowed_values={
        "regulation_mode": {RegulationMode.CONTINUOUS_FEEDBACK, RegulationMode.HYBRID}
    },
    status=ENGINEERING,
)
@dataclass(eq=False)
class FlowRegulationPhase(ActionPhase):
    """
    Continuous modulation of tilt angle and grip and load force while substance leaves
    the container, against the tracked quantity in the receptacle.

    The one phase that is coupled and continuously modulated at once. It is where the
    discrete event-boundary assumption of the core is weakest and continuous feedback
    dominates, so it must monitor at least one signal.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class FlowTerminationPhase(ActionPhase):
    """
    Counter-rotation to bring the substance back below the lip.

    Ends at flow cessation.
    """


@POURING.concept(status=ENGINEERING)
@dataclass(eq=False)
class ReturnUprightPhase(ActionPhase):
    """
    Restoration of the container to a stable orientation after flow has stopped.
    """
