"""
What makes a prehension phase structure admissible.

Whatever the declarations already say is checked without being written here, among them the
coupling a phase fixes, the contact change an event fixes, the kinds of threshold and
setpoint an event may cross or attain, and that a structure declares everything it refers
to. The rules below relate phases to their boundary events and the members of a coupling to
each other. Each names itself with a member of :class:`PrehensionRule`, and the
documentation of that member is the reason the rule exists.
"""

from __future__ import annotations

from enum import StrEnum

from llm_reasoner.knowledge_updated.microworlds.flanagan.core.rules import CORE_RULES
from llm_reasoner.knowledge_updated.microworlds.flanagan.object_dynamics.rules import (
    OBJECT_DYNAMICS_RULES,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.ontology import FLANAGAN
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.control import (
    AnticipatoryGripAdjustment,
    GripLoadCoupling,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.events import (
    ContactBreak,
    ForcesUnloaded,
    Touchdown,
    WithdrawalComplete,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.phases import (
    ContactReleasePhase,
    ReplacePhase,
    UnloadPhase,
    WithdrawPhase,
)
from llm_reasoner.knowledge_updated.microworlds.flanagan.prehension.structure import (
    PrehensionPhaseStructure,
)
from llm_reasoner.knowledge_updated.ontological.acceptance import (
    AcceptanceRule,
    AcceptanceSuite,
    Forbidden,
    Inclusion,
    MostSpecificTypes,
    RequiresKind,
)
from llm_reasoner.knowledge_updated.ontological.derived import DerivedChecks

# %% the rules


class PrehensionRule(StrEnum):
    """
    A rule a prehension phase structure must satisfy.
    """

    GRIP_MARGIN_ON_ANOTHER_SETPOINT = "grip_margin_on_another_setpoint"
    """
    The safety margin of a grip-load coupling must be the margin of the grip force that the
    coupling produces. Otherwise the margin constrains a different force.
    """

    REPLACE_PHASE_NOT_ENDED_BY_TOUCHDOWN = "replace_phase_not_ended_by_touchdown"
    """
    The object stays coupled to the effector while being lowered and transfers its load
    only at touchdown.
    """

    UNLOAD_PHASE_NOT_ENDED_BY_FORCES_UNLOADED = (
        "unload_phase_not_ended_by_forces_unloaded"
    )
    """
    Unloading is one stage of control and releasing contact is another. It ends when
    support transfer and the low forces complete, not when contact breaks.
    """

    RELEASE_PHASE_NOT_ENDED_BY_CONTACT_BREAK = (
        "release_phase_not_ended_by_contact_break"
    )
    """
    Release is individuated by the loss of contact between effector and object.
    """

    WITHDRAWAL_NOT_AFTER_CONTACT_BREAK = "withdrawal_not_after_contact_break"
    """
    Withdrawal follows contact release and is not part of it.
    """

    WITHDRAWAL_NOT_ENDED_BY_COMPLETION = "withdrawal_not_ended_by_completion"
    """
    Withdrawal ends when the effector reaches its withdrawal target.
    """

    ANTICIPATORY_ADJUSTMENT_NOT_FOR_TOUCHDOWN = (
        "anticipatory_adjustment_not_for_touchdown"
    )
    """
    Impact-related grip modulation must be scheduled from a predicted touchdown.
    """

    PHASE_NOT_A_SPECIFIC_KIND = "phase_not_a_specific_kind"
    """
    A prehension phase must be a specific grasping, lifting, transporting, replacing or
    releasing phase, since the general phase says nothing about how control proceeds.
    """


# %% predicates that no declared kind of rule expresses


def adjustment_is_not_for_touchdown(adjustment: AnticipatoryGripAdjustment) -> bool:
    """
    Return whether an adjustment is triggered by a prediction of something other than
    touchdown.
    """
    prediction = adjustment.triggered_by_prediction
    return prediction is not None and not isinstance(
        prediction.predicts_event, Touchdown
    )


# %% the rules

PREHENSION_RULES: tuple[AcceptanceRule, ...] = (
    Inclusion(
        id=PrehensionRule.GRIP_MARGIN_ON_ANOTHER_SETPOINT,
        applies_to=GripLoadCoupling,
        source="safety_margin.margin_on",
        target="grip_force_response",
    ),
    RequiresKind(
        id=PrehensionRule.REPLACE_PHASE_NOT_ENDED_BY_TOUCHDOWN,
        applies_to=ReplacePhase,
        path="terminated_by",
        kind=Touchdown,
    ),
    RequiresKind(
        id=PrehensionRule.UNLOAD_PHASE_NOT_ENDED_BY_FORCES_UNLOADED,
        applies_to=UnloadPhase,
        path="terminated_by",
        kind=ForcesUnloaded,
    ),
    RequiresKind(
        id=PrehensionRule.RELEASE_PHASE_NOT_ENDED_BY_CONTACT_BREAK,
        applies_to=ContactReleasePhase,
        path="terminated_by",
        kind=ContactBreak,
    ),
    RequiresKind(
        id=PrehensionRule.WITHDRAWAL_NOT_AFTER_CONTACT_BREAK,
        applies_to=WithdrawPhase,
        path="initiated_by",
        kind=ContactBreak,
    ),
    RequiresKind(
        id=PrehensionRule.WITHDRAWAL_NOT_ENDED_BY_COMPLETION,
        applies_to=WithdrawPhase,
        path="terminated_by",
        kind=WithdrawalComplete,
    ),
    Forbidden(
        id=PrehensionRule.ANTICIPATORY_ADJUSTMENT_NOT_FOR_TOUCHDOWN,
        applies_to=AnticipatoryGripAdjustment,
        violated_when=adjustment_is_not_for_touchdown,
        message="the adjustment is not triggered by a predicted touchdown",
    ),
    MostSpecificTypes(
        id=PrehensionRule.PHASE_NOT_A_SPECIFIC_KIND,
        applies_to=PrehensionPhaseStructure,
        collection="defines_phase",
    ),
)
"""
Every rule of the module, in the order of the enumeration that names them.
"""

PREHENSION_ACCEPTANCE = AcceptanceSuite(
    name="flanagan-prehension",
    ontology=FLANAGAN,
    rules=(*CORE_RULES, *OBJECT_DYNAMICS_RULES, *PREHENSION_RULES),
    checks=(DerivedChecks(FLANAGAN),),
)
"""
The checks that decide whether a prehension phase structure is admissible.
"""
