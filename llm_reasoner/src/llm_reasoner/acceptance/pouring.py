"""
Acceptance rules of the pouring module: which phases and events a pouring structure has,
in what order, and which flow dynamics its material admits.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import Sequence

from llm_reasoner.acceptance.rules import Severity, StructureRule
from llm_reasoner.ontologies.predictive_manipulation import core, pouring, prehension

# %% rule names


class PouringRule(StrEnum):
    """
    The acceptance rules of the pouring module.
    """

    POURING_EVENTS_SPECIFIC = "pouring_events_specific"
    """
    A pouring structure classifies every event by its most specific kind; a generic
    event type discards what the vocabulary says about flow, contact and support.
    """

    POURING_PHASES_COVERED = "pouring_phases_covered"
    """
    A pouring structure tilts until flow begins, regulates the flow and terminates it;
    grasping, transport, alignment and returning are optional.
    """

    POURING_PHASES_ORDERED = "pouring_phases_ordered"
    """
    Every transition of a pouring structure follows the sensorimotor order of pouring;
    omitted phases are allowed, reversed ones are not.
    """

    MATERIAL_FITS_FLOW_DYNAMICS = "material_fits_flow_dynamics"
    """
    Each material admits only its own flow dynamics: a liquid does not flow like a
    granular or cohesive material.
    """

    TERMINATION_REASON_HAS_EVENT = "termination_reason_has_event"
    """
    The reason a transfer stops names an event the structure can predict and detect.
    """


# %% vocabulary


GENERIC_EVENT_TYPES: tuple[type, ...] = (
    core.PhaseBoundaryEventType,
    core.MechanicalEventType,
    core.ContactEventType,
    core.ThresholdCrossingEventType,
    core.SetpointAttainmentEventType,
    core.PoseAttainmentEventType,
    core.FailureEventType,
)
"""
The event types that only name a family of events, never a particular one.
"""

POURING_ORDER: tuple[type, ...] = (
    prehension.ReachPhase,
    prehension.ContactEstablishmentPhase,
    prehension.ForceLoadingPhase,
    prehension.LiftPhase,
    prehension.HoldPhase,
    prehension.TransportPhase,
    pouring.AlignmentPhase,
    pouring.TiltToFlowPhase,
    pouring.FlowRegulationPhase,
    pouring.FlowTerminationPhase,
    pouring.ReturnUprightPhase,
    prehension.TransportPhase,
    prehension.ReplacePhase,
    prehension.UnloadPhase,
    prehension.ContactReleasePhase,
    prehension.WithdrawPhase,
)
"""
The order of the phases of pouring. Transport appears twice: carrying the source to
the target, and carrying it back.
"""

REQUIRED_PHASES: tuple[type, ...] = (
    pouring.TiltToFlowPhase,
    pouring.FlowRegulationPhase,
    pouring.FlowTerminationPhase,
)
"""
The phases every pouring structure has.
"""

EXCLUDED_DYNAMICS: dict[pouring.MaterialRegime, tuple[type, ...]] = {
    pouring.MaterialRegime.LIQUID: (
        pouring.GranularFlowDynamicsEstimate,
        pouring.CohesiveFlowDynamicsEstimate,
    ),
    pouring.MaterialRegime.GRANULAR: (
        pouring.DynamicViscosityEstimate,
        pouring.SloshDynamicsEstimate,
        pouring.CohesiveFlowDynamicsEstimate,
    ),
    pouring.MaterialRegime.COHESIVE: (
        pouring.SloshDynamicsEstimate,
        pouring.GranularFlowDynamicsEstimate,
    ),
}
"""
The flow dynamics estimates each material regime rules out.
"""

TERMINATION_EVENTS: dict[pouring.TransferTerminationReason, type] = {
    pouring.TransferTerminationReason.TARGET_QUANTITY_ATTAINED: pouring.TargetQuantityReached,
    pouring.TransferTerminationReason.SOURCE_EXHAUSTED: pouring.SourceEmpty,
    pouring.TransferTerminationReason.INSTRUCTED_STOP: core.ExternalTerminationEventType,
    pouring.TransferTerminationReason.TRANSFER_FAILURE: core.FailureEventType,
}
"""
The type of event that detects each reason for a transfer to stop.
"""


def positions(phase: core.ActionPhase, order: Sequence[type]) -> list[int]:
    """
    :return: The positions in the order a phase can take.
    """
    return [position for position, kind in enumerate(order) if isinstance(phase, kind)]


# %% rules


class PouringEventsSpecific(StructureRule[pouring.PouringPhaseStructure]):
    """
    No event of a pouring structure is of a generic event type.
    """

    @property
    def name(self) -> StrEnum:
        return PouringRule.POURING_EVENTS_SPECIFIC

    @property
    def reason(self) -> str:
        return "an event is of a generic event type rather than a specific one"

    def involved(
        self, fragment: pouring.PouringPhaseStructure
    ) -> list[core.PhaseBoundaryEventType]:
        """
        :return: The events of a generic event type.
        """
        return [
            event
            for event in fragment.defines_event_type
            if type(event) in GENERIC_EVENT_TYPES
        ]

    def is_violated_by(self, fragment: pouring.PouringPhaseStructure) -> bool:
        return bool(self.involved(fragment))


class PouringPhasesCovered(StructureRule[pouring.PouringPhaseStructure]):
    """
    A pouring structure has a tilt-to-flow, a flow regulation and a flow termination
    phase.
    """

    @property
    def name(self) -> StrEnum:
        return PouringRule.POURING_PHASES_COVERED

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def reason(self) -> str:
        return "the tilt-to-flow, flow regulation or flow termination phase is missing"

    def is_violated_by(self, fragment: pouring.PouringPhaseStructure) -> bool:
        return not all(
            any(isinstance(phase, required) for phase in fragment.defines_phase)
            for required in REQUIRED_PHASES
        )


class PouringPhasesOrdered(StructureRule[pouring.PouringPhaseStructure]):
    """
    Every transition of a pouring structure leads to a phase that can follow its
    source in the order of pouring.
    """

    @property
    def name(self) -> StrEnum:
        return PouringRule.POURING_PHASES_ORDERED

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def reason(self) -> str:
        return "a transition leads back against the order of pouring"

    def is_violated_by(self, fragment: pouring.PouringPhaseStructure) -> bool:
        for transition in fragment.defines_transition:
            if transition.from_phase is None or transition.to_phase is None:
                continue
            sources = positions(transition.from_phase, POURING_ORDER)
            targets = positions(transition.to_phase, POURING_ORDER)
            if (
                sources
                and targets
                and not any(source < target for source in sources for target in targets)
            ):
                return True
        return False


class MaterialFitsFlowDynamics(StructureRule[pouring.PouringPhaseStructure]):
    """
    A pouring structure declares no flow dynamics its material rules out.
    """

    @property
    def name(self) -> StrEnum:
        return PouringRule.MATERIAL_FITS_FLOW_DYNAMICS

    @property
    def reason(self) -> str:
        return "declares flow dynamics its material regime rules out"

    def is_violated_by(self, fragment: pouring.PouringPhaseStructure) -> bool:
        excluded = EXCLUDED_DYNAMICS.get(fragment.has_material_regime, ())
        return any(
            isinstance(parameter, excluded) for parameter in fragment.defines_parameter
        )


class TerminationReasonHasEvent(StructureRule[pouring.PouringPhaseStructure]):
    """
    A pouring structure declares an event detecting the reason its transfer stops.
    """

    @property
    def name(self) -> StrEnum:
        return PouringRule.TERMINATION_REASON_HAS_EVENT

    @property
    def reason(self) -> str:
        return "declares no event detecting has_termination_reason"

    def is_violated_by(self, fragment: pouring.PouringPhaseStructure) -> bool:
        if fragment.has_termination_reason is None:
            return False
        detecting = TERMINATION_EVENTS[fragment.has_termination_reason]
        return not any(
            isinstance(event, detecting) for event in fragment.defines_event_type
        )
