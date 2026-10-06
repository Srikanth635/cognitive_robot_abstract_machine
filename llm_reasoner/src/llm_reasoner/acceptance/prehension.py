"""
Acceptance rules of the prehension module: grip and load control, and the boundaries of
the phases that put an object down and let go of it.
"""

from __future__ import annotations

from enum import StrEnum

from krrood.entity_query_language.factories import and_, contains, flat_variable
from krrood.entity_query_language.predicate import HasType
from typing_extensions import Any

from llm_reasoner.acceptance.rules import FragmentRule, RequirementRule, StructureRule
from llm_reasoner.ontologies.predictive_manipulation import core, prehension

# %% rule names


class PrehensionRule(StrEnum):
    """
    The acceptance rules of the prehension module.
    """

    SAFETY_MARGIN_ON_COUPLING_RESPONSE = "safety_margin_on_coupling_response"
    """
    A grip-load coupling's safety margin is kept on the grip force the coupling
    produces, not on another one.
    """

    REPLACE_ENDS_AT_TOUCHDOWN = "replace_ends_at_touchdown"
    """
    Lowering an object ends when it touches its support, where its load starts to pass
    to the support.
    """

    UNLOAD_ENDS_WHEN_FORCES_UNLOADED = "unload_ends_when_forces_unloaded"
    """
    Unloading ends once grip and load forces are low and the support bears the weight;
    releasing contact is a separate stage.
    """

    RELEASE_ENDS_AT_GRASP_RELEASE = "release_ends_at_grasp_release"
    """
    Releasing an object ends when the effector loses contact with it.
    """

    WITHDRAWAL_FOLLOWS_RELEASE = "withdrawal_follows_release"
    """
    Withdrawing the effector begins at release and ends at its withdrawal target; it
    follows releasing rather than being part of it.
    """

    ADJUSTMENT_PRECEDES_PREDICTED_EVENT = "adjustment_precedes_predicted_event"
    """
    An anticipatory grip adjustment is active in a phase that ends at the event its
    triggering prediction is about, so the grip changes before that event, not after.
    """

    PREHENSION_PHASES_SPECIFIC = "prehension_phases_specific"
    """
    A prehension structure classifies every phase by its most specific kind; a plain
    phase discards what the prehension vocabulary says about grasping, lifting,
    transporting, replacing or releasing.
    """


# %% grip and load


class SafetyMarginOnCouplingResponse(FragmentRule[prehension.GripLoadCoupling]):
    """
    A coupling's safety margin is kept on the coupling's grip-force response.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.SAFETY_MARGIN_ON_COUPLING_RESPONSE

    @property
    def reason(self) -> str:
        return (
            "has_safety_margin is kept on another grip force than grip_force_response"
        )

    def violating(self, subject: Any) -> Any:
        return and_(
            subject.has_safety_margin != None,
            subject.grip_force_response != None,
            subject.has_safety_margin.margin_on != subject.grip_force_response,
        )


class AdjustmentPrecedesPredictedEvent(
    RequirementRule[prehension.AnticipatoryGripAdjustment]
):
    """
    An anticipatory adjustment is active in a phase ending at the event its triggering
    prediction concerns.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.ADJUSTMENT_PRECEDES_PREDICTED_EVENT

    @property
    def reason(self) -> str:
        return "adjustment_active_in names no phase ending at the predicted event"

    def applies(self, subject: Any) -> Any:
        return and_(
            subject.triggered_by_prediction != None,
            subject.triggered_by_prediction.caused_by_event != None,
        )

    def meets(self, subject: Any) -> Any:
        phase = flat_variable(subject.adjustment_active_in)
        return contains(
            phase.terminated_by, subject.triggered_by_prediction.caused_by_event
        )


# %% phase boundaries


class ReplaceEndsAtTouchdown(RequirementRule[prehension.ReplacePhase]):
    """
    A replace phase is terminated by a touchdown.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.REPLACE_ENDS_AT_TOUCHDOWN

    @property
    def reason(self) -> str:
        return "terminated_by names no touchdown"

    def applies(self, subject: Any) -> Any:
        return HasType(subject, prehension.ReplacePhase)

    def meets(self, subject: Any) -> Any:
        return HasType(flat_variable(subject.terminated_by), prehension.Touchdown)


class UnloadEndsWhenForcesUnloaded(RequirementRule[prehension.UnloadPhase]):
    """
    An unload phase is terminated when forces are unloaded.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.UNLOAD_ENDS_WHEN_FORCES_UNLOADED

    @property
    def reason(self) -> str:
        return "terminated_by names no forces-unloaded event"

    def applies(self, subject: Any) -> Any:
        return HasType(subject, prehension.UnloadPhase)

    def meets(self, subject: Any) -> Any:
        return HasType(flat_variable(subject.terminated_by), prehension.ForcesUnloaded)


class ReleaseEndsAtGraspRelease(RequirementRule[prehension.ContactReleasePhase]):
    """
    A contact release phase is terminated by a grasp release.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.RELEASE_ENDS_AT_GRASP_RELEASE

    @property
    def reason(self) -> str:
        return "terminated_by names no grasp release"

    def applies(self, subject: Any) -> Any:
        return HasType(subject, prehension.ContactReleasePhase)

    def meets(self, subject: Any) -> Any:
        return HasType(flat_variable(subject.terminated_by), prehension.GraspRelease)


class WithdrawalFollowsRelease(RequirementRule[prehension.WithdrawPhase]):
    """
    A withdraw phase is initiated by a grasp release and terminated by reaching its
    withdrawal target.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.WITHDRAWAL_FOLLOWS_RELEASE

    @property
    def reason(self) -> str:
        return "not initiated by a grasp release or not ending at the withdrawal target"

    def applies(self, subject: Any) -> Any:
        return HasType(subject, prehension.WithdrawPhase)

    def meets(self, subject: Any) -> Any:
        return and_(
            HasType(flat_variable(subject.initiated_by), prehension.GraspRelease),
            HasType(
                flat_variable(subject.terminated_by), prehension.WithdrawalComplete
            ),
        )


# %% classification


class PrehensionPhasesSpecific(StructureRule[prehension.PrehensionPhaseStructure]):
    """
    Every phase of a prehension structure is of a kind more specific than a plain
    phase.
    """

    @property
    def name(self) -> StrEnum:
        return PrehensionRule.PREHENSION_PHASES_SPECIFIC

    @property
    def reason(self) -> str:
        return "a phase is a plain ActionPhase rather than a specific kind of phase"

    def is_violated_by(self, fragment: prehension.PrehensionPhaseStructure) -> bool:
        return any(type(phase) is core.ActionPhase for phase in fragment.defines_phase)
