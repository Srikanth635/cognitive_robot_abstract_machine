"""
Acceptance constraints for action-neutral prehensile manipulation.
"""

from __future__ import annotations

from llm_reasoner.knowledge.microworlds.flanagan.core import (
    ContactEventType,
    ContactTransition,
    CouplingRegime,
)
from llm_reasoner.knowledge.microworlds.flanagan.prehension import (
    AnticipatoryGripAdjustment,
    ContactBreak,
    ContactEstablishment,
    ContactReleasePhase,
    CorrectiveGripResponse,
    ForcesUnloaded,
    GripForceSetpoint,
    GripLoadCoupling,
    Liftoff,
    PrehensionPhaseStructure,
    ReplacePhase,
    Touchdown,
    UnloadPhase,
    WithdrawalComplete,
    WithdrawPhase,
)
from llm_reasoner.knowledge.ontological.constraints import (
    Closure,
    Forbidden,
    MostSpecificTypes,
)

# %% coherence predicates


def coupling_has_unrelated_safety_margin(coupling: GripLoadCoupling) -> bool:
    """
    Return whether the margin does not constrain the coupling response.
    """
    if coupling.safety_margin is None or coupling.grip_force_response is None:
        return False
    return coupling.safety_margin.margin_on is not coupling.grip_force_response


def replace_phase_decouples_before_touchdown(phase: ReplacePhase) -> bool:
    """
    Return whether lowering is not coupled through its touchdown boundary.
    """
    return phase.coupling_regime is not CouplingRegime.COUPLED or not any(
        isinstance(event, Touchdown) for event in phase.terminated_by
    )


def unload_phase_lacks_force_completion(phase: UnloadPhase) -> bool:
    """
    Return whether unloading ends without a low-force boundary.
    """
    return not any(isinstance(event, ForcesUnloaded) for event in phase.terminated_by)


def contact_release_lacks_contact_break(phase: ContactReleasePhase) -> bool:
    """
    Return whether contact release ends without breaking contact.
    """
    return not any(isinstance(event, ContactBreak) for event in phase.terminated_by)


def withdrawal_has_incoherent_boundaries(phase: WithdrawPhase) -> bool:
    """
    Return whether optional withdrawal is not bounded after contact release.
    """
    return not any(
        isinstance(event, ContactBreak) for event in phase.initiated_by
    ) or not any(isinstance(event, WithdrawalComplete) for event in phase.terminated_by)


def corrective_grip_response_targets_other_setpoints(
    response: CorrectiveGripResponse,
) -> bool:
    """
    Return whether a grip response targets a non-grip setpoint.
    """
    return any(
        not isinstance(setpoint, GripForceSetpoint)
        for setpoint in response.targets_setpoint
    )


def anticipatory_adjustment_lacks_touchdown_context(
    adjustment: AnticipatoryGripAdjustment,
) -> bool:
    """
    Return whether an anticipatory impact response lacks touchdown context.
    """
    prediction = adjustment.triggered_by_prediction
    if prediction is None:
        return False
    return not isinstance(prediction.predicts_event, Touchdown) or any(
        not isinstance(phase, ReplacePhase) for phase in adjustment.adjustment_active_in
    )


def contact_event_has_wrong_transition(event: ContactEventType) -> bool:
    """
    Return whether a typed contact event declares the opposite topology change.
    """
    if isinstance(event, (ContactEstablishment, Touchdown)):
        return event.contact_transition is not ContactTransition.ESTABLISHES
    if isinstance(event, (ContactBreak, Liftoff)):
        return event.contact_transition is not ContactTransition.BREAKS
    return False


# %% acceptance constraints


PREHENSION_CONSTRAINTS = [
    Forbidden(
        id="PRE-C59",
        applies_to=GripLoadCoupling,
        when=coupling_has_unrelated_safety_margin,
        message="grip-load safety margin constrains a different force response",
        rationale="The declared safety margin must be part of the same functional "
        "relationship as the resulting grip-force response.",
    ),
    Forbidden(
        id="PRE-C60",
        applies_to=ReplacePhase,
        when=replace_phase_decouples_before_touchdown,
        message="lowering is not coupled through a touchdown boundary",
        rationale="The object remains mechanically coupled to the effector while "
        "being lowered and only transfers load at touchdown.",
    ),
    Forbidden(
        id="PRE-C61",
        applies_to=UnloadPhase,
        when=unload_phase_lacks_force_completion,
        message="unloading does not end when support transfer and low forces complete",
        rationale="Force unloading and contact release are separate control stages.",
    ),
    Forbidden(
        id="PRE-C62",
        applies_to=ContactReleasePhase,
        when=contact_release_lacks_contact_break,
        message="contact release does not end at effector-object contact break",
        rationale="Contact release is individuated by loss of effector-object contact.",
    ),
    Forbidden(
        id="PRE-C63",
        applies_to=WithdrawPhase,
        when=withdrawal_has_incoherent_boundaries,
        message="withdrawal is not bounded after contact break",
        rationale="Optional withdrawal follows contact release rather than being part "
        "of it.",
    ),
    Forbidden(
        id="PRE-C64",
        applies_to=CorrectiveGripResponse,
        when=corrective_grip_response_targets_other_setpoints,
        message="corrective grip response targets a non-grip setpoint",
        rationale="A typed grip response must change grip force; other corrections "
        "remain generic corrective actions.",
    ),
    Forbidden(
        id="PRE-C65",
        applies_to=AnticipatoryGripAdjustment,
        when=anticipatory_adjustment_lacks_touchdown_context,
        message=(
            "anticipatory impact grip lacks a touchdown prediction or replace phase"
        ),
        rationale="Impact-related grip modulation must be scheduled before the "
        "predicted touchdown while the object is being lowered.",
    ),
    Forbidden(
        id="PRE-C66",
        applies_to=ContactEventType,
        when=contact_event_has_wrong_transition,
        message="typed contact event declares the opposite contact transition",
        rationale="Contact establishment and touchdown create contact, while contact "
        "break and liftoff remove it.",
    ),
    Closure(
        id="PRE-C13v",
        applies_to=PrehensionPhaseStructure,
        container="defines_parameter",
        reached_via=(
            "has_force_coupling.predicted_load",
            "has_force_coupling.friction_estimate",
            "has_force_coupling.safety_margin",
            "has_force_coupling.grip_force_response",
            "has_anticipatory_grip_adjustment.targets_grip_force",
        ),
        rationale="Every parameter in prehensile force control must be declared by "
        "its structure.",
    ),
    Closure(
        id="PRE-C13w",
        applies_to=PrehensionPhaseStructure,
        container="defines_prediction",
        reached_via=("has_anticipatory_grip_adjustment.triggered_by_prediction",),
        rationale="Every prediction scheduling anticipatory grip must be declared.",
    ),
    Closure(
        id="PRE-C13x",
        applies_to=PrehensionPhaseStructure,
        container="defines_phase",
        reached_via=(
            "has_force_coupling.coupling_active_in",
            "has_anticipatory_grip_adjustment.adjustment_active_in",
        ),
        rationale="Every phase carrying prehensile force control must be declared.",
    ),
    Closure(
        id="PRE-C13y",
        applies_to=PrehensionPhaseStructure,
        container="defines_role",
        reached_via=("defines_event_type.weight_supported_by",),
        rationale="Every support receiving manipulandum weight must be declared.",
    ),
    MostSpecificTypes(
        id="PRE-C30d",
        applies_to=PrehensionPhaseStructure,
        collection="defines_phase",
        rationale="A prehension phase requires a specific grasping, lifting, "
        "transport, replacement, or release concept.",
    ),
]
