"""
Acceptance rules of the prehension module: grip and load control, and the boundaries of
the phases that put an object down and let go of it.
"""

from __future__ import annotations

from enum import StrEnum

from krrood.entity_query_language.factories import and_, contains, flat_variable
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
