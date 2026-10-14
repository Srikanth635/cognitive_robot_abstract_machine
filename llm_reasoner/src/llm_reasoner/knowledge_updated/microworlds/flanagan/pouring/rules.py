"""
What makes a pouring phase structure admissible.

Whatever the declarations already say is checked without being written here, among them
the coupling and regulation of the flow phase, the dimensions of quantities and rates, the
kinds of threshold and setpoint that events name, and that a structure declares everything
it refers to. The continuous control of a phase is checked by the rules of the core. The
rules below relate the material and the reason for stopping to the rest of the structure,
and the phases to their events and their order. Each names itself with a member of
:class:`PouringRule`, and the documentation of that member is the reason the rule exists.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Callable, Mapping

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.events import (
    ExternalTerminationEventType,
    PhaseBoundaryEventType,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.core.rules import CORE_RULES
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.rules import (
    OBJECT_DYNAMICS_RULES,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import FLANAGAN
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.events import (
    FlowCessation,
    FlowOnset,
    SourceEmpty,
    TargetQuantityReached,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.parameters import (
    CohesiveFlowDynamicsEstimate,
    DynamicViscosityEstimate,
    GranularFlowDynamicsEstimate,
    SloshDynamicsEstimate,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.phases import (
    AlignmentPhase,
    FlowRegulationPhase,
    FlowTerminationPhase,
    ReturnUprightPhase,
    TiltToFlowPhase,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.structure import (
    PouringPhaseStructure,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.pouring.value_partitions import (
    MaterialRegime,
    TransferTerminationReason,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.phases import (
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
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.rules import (
    PREHENSION_RULES,
)
from llm_reasoner.knowledge_updated.ontological.acceptance import (
    AcceptanceRule,
    AcceptanceSuite,
    Forbidden,
    MostSpecificTypes,
    RequiresKind,
    TypeOrder,
)
from llm_reasoner.knowledge_updated.ontological.derived import DerivedChecks
from llm_reasoner.knowledge_updated.ontological.vocabulary import Severity

# %% what a material admits and what a stop reason needs

MATERIALS_OF_ESTIMATE: Mapping[type, frozenset[MaterialRegime]] = {
    DynamicViscosityEstimate: frozenset(
        {MaterialRegime.LIQUID, MaterialRegime.COHESIVE}
    ),
    SloshDynamicsEstimate: frozenset({MaterialRegime.LIQUID}),
    GranularFlowDynamicsEstimate: frozenset({MaterialRegime.GRANULAR}),
    CohesiveFlowDynamicsEstimate: frozenset({MaterialRegime.COHESIVE}),
}
"""
The materials each estimate of flow behaviour is physically relevant to.
"""

EVIDENCE_OF_STOPPING: Mapping[
    TransferTerminationReason, Callable[[PhaseBoundaryEventType], bool]
] = {
    TransferTerminationReason.TARGET_QUANTITY: lambda event: isinstance(
        event, TargetQuantityReached
    ),
    TransferTerminationReason.SOURCE_EMPTY: lambda event: isinstance(
        event, SourceEmpty
    ),
    TransferTerminationReason.EXPLICIT_INSTRUCTION: lambda event: isinstance(
        event, ExternalTerminationEventType
    ),
    TransferTerminationReason.FAILURE: lambda event: event.is_failure_event,
}
"""
For each reason for stopping, what recognises an event the structure can detect it by.
"""

# %% the rules


class PouringRule(StrEnum):
    """
    A rule a pouring phase structure must satisfy.
    """

    MATERIAL_REGIME_EXCLUDES_ESTIMATE = "material_regime_excludes_estimate"
    """
    Each material admits only the flow behaviours that are physically relevant to it, so a
    structure cannot estimate the slosh of grains or the viscosity of a granular flow.
    """

    TERMINATION_REASON_HAS_NO_EVENT = "termination_reason_has_no_event"
    """
    A reason for stopping the transfer must identify an event that the structure can
    predict and detect.
    """

    TRANSFER_HAS_NO_TILT_TO_FLOW_PHASE = "transfer_has_no_tilt_to_flow_phase"
    """
    A pouring structure needs the phase in which flow begins.
    """

    TRANSFER_HAS_NO_FLOW_REGULATION_PHASE = "transfer_has_no_flow_regulation_phase"
    """
    A pouring structure needs the phase in which flow is regulated.
    """

    TRANSFER_HAS_NO_FLOW_TERMINATION_PHASE = "transfer_has_no_flow_termination_phase"
    """
    A pouring structure needs the phase in which flow is stopped. Grasping, aligning,
    transporting and returning upright remain optional.
    """

    PHASES_AGAINST_SENSORIMOTOR_ORDER = "phases_against_sensorimotor_order"
    """
    Every phase that is present must follow the sensorimotor order. Phases that are left
    out do not make a structure incomplete.
    """

    EVENT_NOT_A_SPECIFIC_KIND = "event_not_a_specific_kind"
    """
    A boundary event should be of the most specific pouring or manipulation kind
    available, since a general event discards what queries and displays rely on.
    """

    TILT_PHASE_NOT_ENDED_BY_FLOW_ONSET = "tilt_phase_not_ended_by_flow_onset"
    """
    Tilting lasts until the substance begins to flow.
    """

    FLOW_TERMINATION_NOT_ENDED_BY_FLOW_CESSATION = (
        "flow_termination_not_ended_by_flow_cessation"
    )
    """
    Stopping the flow lasts until the substance no longer crosses the lip.
    """


# %% predicates that no declared kind of rule expresses


def structure_has_estimate_of_another_material(
    structure: PouringPhaseStructure,
) -> bool:
    """
    Return whether a declared estimate is not relevant to the material being poured.
    """
    return any(
        isinstance(parameter, kind) and structure.material_regime not in materials
        for parameter in structure.defines_parameter
        for kind, materials in MATERIALS_OF_ESTIMATE.items()
    )


def structure_stops_for_a_reason_no_event_shows(
    structure: PouringPhaseStructure,
) -> bool:
    """
    Return whether no declared event is evidence of the reason for stopping.
    """
    shows = EVIDENCE_OF_STOPPING.get(structure.termination_reason)
    return shows is not None and not any(
        shows(event) for event in structure.defines_event_type
    )


# %% the rules

SENSORIMOTOR_ORDER = (
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
)
"""
The kinds of phase in the order a manipulation passes through them.
"""

POURING_RULES: tuple[AcceptanceRule, ...] = (
    Forbidden(
        id=PouringRule.MATERIAL_REGIME_EXCLUDES_ESTIMATE,
        applies_to=PouringPhaseStructure,
        violated_when=structure_has_estimate_of_another_material,
        message="a declared estimate is not relevant to the material being poured",
    ),
    Forbidden(
        id=PouringRule.TERMINATION_REASON_HAS_NO_EVENT,
        applies_to=PouringPhaseStructure,
        violated_when=structure_stops_for_a_reason_no_event_shows,
        message="no declared event shows the reason for stopping the transfer",
    ),
    RequiresKind(
        id=PouringRule.TRANSFER_HAS_NO_TILT_TO_FLOW_PHASE,
        applies_to=PouringPhaseStructure,
        severity=Severity.WARNING,
        path="defines_phase",
        kind=TiltToFlowPhase,
    ),
    RequiresKind(
        id=PouringRule.TRANSFER_HAS_NO_FLOW_REGULATION_PHASE,
        applies_to=PouringPhaseStructure,
        severity=Severity.WARNING,
        path="defines_phase",
        kind=FlowRegulationPhase,
    ),
    RequiresKind(
        id=PouringRule.TRANSFER_HAS_NO_FLOW_TERMINATION_PHASE,
        applies_to=PouringPhaseStructure,
        severity=Severity.WARNING,
        path="defines_phase",
        kind=FlowTerminationPhase,
    ),
    TypeOrder(
        id=PouringRule.PHASES_AGAINST_SENSORIMOTOR_ORDER,
        applies_to=PouringPhaseStructure,
        severity=Severity.WARNING,
        edges="defines_transition",
        tail="from_phase",
        head="to_phase",
        expected_types=SENSORIMOTOR_ORDER,
    ),
    MostSpecificTypes(
        id=PouringRule.EVENT_NOT_A_SPECIFIC_KIND,
        applies_to=PouringPhaseStructure,
        collection="defines_event_type",
    ),
    RequiresKind(
        id=PouringRule.TILT_PHASE_NOT_ENDED_BY_FLOW_ONSET,
        applies_to=TiltToFlowPhase,
        path="terminated_by",
        kind=FlowOnset,
    ),
    RequiresKind(
        id=PouringRule.FLOW_TERMINATION_NOT_ENDED_BY_FLOW_CESSATION,
        applies_to=FlowTerminationPhase,
        path="terminated_by",
        kind=FlowCessation,
    ),
)
"""
Every rule of the module, in the order of the enumeration that names them.
"""

POURING_ACCEPTANCE = AcceptanceSuite(
    name="flanagan-pouring",
    ontology=FLANAGAN,
    rules=(*CORE_RULES, *OBJECT_DYNAMICS_RULES, *PREHENSION_RULES, *POURING_RULES),
    checks=(DerivedChecks(FLANAGAN),),
)
"""
The checks that decide whether a pouring phase structure is admissible.
"""
