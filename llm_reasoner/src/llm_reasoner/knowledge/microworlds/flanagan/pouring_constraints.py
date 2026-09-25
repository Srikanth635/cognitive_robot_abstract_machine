"""
Acceptance constraints owned by the substance-transfer extension.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ExternalTerminationEventType,
)
from llm_reasoner.knowledge.microworlds.flanagan.pouring import (
    AlignmentPhase,
    CohesiveFlowDynamicsEstimate,
    DynamicViscosityEstimate,
    FlowRegulationPhase,
    FlowTerminationPhase,
    GranularFlowDynamicsEstimate,
    MaterialRegime,
    PouringPhaseStructure,
    ReturnUprightPhase,
    SloshDynamicsEstimate,
    SourceEmpty,
    TargetQuantityReached,
    TiltToFlowPhase,
    TransferTerminationReason,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    ContactEstablishmentPhase,
    ContactReleasePhase,
    ForceLoadingPhase,
    HoldPhase,
    LiftPhase,
    ReachPhase,
    ReplacePhase,
    TransportPhase,
    UnloadPhase,
    WithdrawPhase,
)
from llm_reasoner.knowledge.ontological.constraints import (
    Forbidden,
    MostSpecificTypes,
    TypeCoverage,
    TypeOrder,
)

# %% material compatibility


def structure_has_incompatible_material_dynamics(
    structure: PouringPhaseStructure,
) -> bool:
    """
    Return whether the structure declares dynamics from another regime.
    """
    parameters = structure.defines_parameter
    has_viscosity = any(
        isinstance(parameter, DynamicViscosityEstimate) for parameter in parameters
    )
    has_slosh = any(
        isinstance(parameter, SloshDynamicsEstimate) for parameter in parameters
    )
    has_granular_dynamics = any(
        isinstance(parameter, GranularFlowDynamicsEstimate) for parameter in parameters
    )
    has_cohesive_dynamics = any(
        isinstance(parameter, CohesiveFlowDynamicsEstimate) for parameter in parameters
    )
    if structure.material_regime is MaterialRegime.LIQUID:
        return has_granular_dynamics or has_cohesive_dynamics
    if structure.material_regime is MaterialRegime.GRANULAR:
        return has_viscosity or has_slosh or has_cohesive_dynamics
    if structure.material_regime is MaterialRegime.COHESIVE:
        return has_slosh or has_granular_dynamics
    return False


def structure_has_mismatched_termination_event(
    structure: PouringPhaseStructure,
) -> bool:
    """
    Return whether no declared boundary event supports the stop reason.
    """
    events = structure.defines_event_type
    if structure.termination_reason is TransferTerminationReason.TARGET_QUANTITY:
        return not any(isinstance(event, TargetQuantityReached) for event in events)
    if structure.termination_reason is TransferTerminationReason.SOURCE_EMPTY:
        return not any(isinstance(event, SourceEmpty) for event in events)
    if structure.termination_reason is TransferTerminationReason.EXPLICIT_INSTRUCTION:
        return not any(
            isinstance(event, ExternalTerminationEventType) for event in events
        )
    if structure.termination_reason is TransferTerminationReason.FAILURE:
        return not any(event.is_failure_event for event in events)
    return False


# %% pouring constraints


POURING_CONSTRAINTS = [
    MostSpecificTypes(
        id="C30a",
        applies_to=PouringPhaseStructure,
        collection="defines_phase",
        rationale="A pouring phase classified only as ActionPhase discards the "
        "action-specific ontology concept needed for querying and display.",
    ),
    MostSpecificTypes(
        id="C30b",
        applies_to=PouringPhaseStructure,
        collection="defines_event_type",
        rationale="A boundary event should use the most specific available pouring "
        "or manipulation event class.",
    ),
    TypeCoverage(
        id="C31",
        applies_to=PouringPhaseStructure,
        severity="warning",
        collection="defines_phase",
        required_types=(
            TiltToFlowPhase,
            FlowRegulationPhase,
            FlowTerminationPhase,
        ),
        rationale="A pouring structure requires transfer onset, regulation, and "
        "termination; prehension, alignment, transport, and return remain optional.",
    ),
    TypeOrder(
        id="C32",
        applies_to=PouringPhaseStructure,
        severity="warning",
        edges="has_transition",
        tail="from_phase",
        head="to_phase",
        expected_types=(
            ReachPhase,
            ContactEstablishmentPhase,
            ForceLoadingPhase,
            LiftPhase,
            HoldPhase,
            TransportPhase,
            AlignmentPhase,
            TiltToFlowPhase,
            FlowRegulationPhase,
            FlowTerminationPhase,
            ReturnUprightPhase,
            ReplacePhase,
            UnloadPhase,
            ContactReleasePhase,
            WithdrawPhase,
        ),
        rationale="Every phase that is present must follow the sensorimotor order; "
        "optional omitted phases do not create coverage failures.",
    ),
    Forbidden(
        id="C57",
        applies_to=PouringPhaseStructure,
        when=structure_has_incompatible_material_dynamics,
        message="material regime is incompatible with the declared flow dynamics",
        rationale="Each material regime admits only its physically relevant flow "
        "dynamics.",
    ),
    Forbidden(
        id="C58",
        applies_to=PouringPhaseStructure,
        when=structure_has_mismatched_termination_event,
        message="termination reason has no corresponding declared boundary event",
        rationale="A transfer stop reason must identify an event the structure can "
        "predict and detect.",
    ),
]


__all__ = ["POURING_CONSTRAINTS"]
