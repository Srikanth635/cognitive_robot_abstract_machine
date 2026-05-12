"""Flanagan-specific view facade over sg_model objects."""

from __future__ import annotations

from dataclasses import dataclass

from llmr_updated_arch.hypotheses.builders.flanagan import FLANAGAN_REASONER_NAME
from llmr_updated_arch.hypotheses.entities.flanagan import (
    FailureModeClaim,
    ForceDynamicEvent,
    GoalConditionClaim,
    PhaseClaim,
    PlanClaim,
    PreconditionClaim,
    RecoveryStrategyClaim,
)
from llmr_updated_arch.hypotheses.views.base import ReasonerGraphView


@dataclass(frozen=True)
class FlanaganGraphView(ReasonerGraphView):
    """Typed query facade for Flanagan motion-plan hypothesis objects."""

    REASONER_NAME = FLANAGAN_REASONER_NAME
    ROOT_CLAIM_TYPES = (PlanClaim,)
    CLAIM_TYPES = (
        PlanClaim,
        PhaseClaim,
        ForceDynamicEvent,
        PreconditionClaim,
        GoalConditionClaim,
        FailureModeClaim,
        RecoveryStrategyClaim,
    )

    def plans(self) -> list[PlanClaim]:
        """Return Flanagan motion-plan claims preserving repository insertion order."""

        return [
            plan
            for plan in self.graph.domain(PlanClaim)
            if plan.meta.source_reasoner == self.REASONER_NAME
        ]

    def phases(self) -> list[PhaseClaim]:
        """Return Flanagan motion-phase claims preserving repository insertion order."""

        return [
            phase
            for phase in self.graph.domain(PhaseClaim)
            if phase.meta.source_reasoner == self.REASONER_NAME
        ]

    def phases_by_name(self, phase_name: str) -> list[PhaseClaim]:
        """Return motion-phase claims whose canonical name equals *phase_name*."""

        return [phase for phase in self.phases() if phase.phase_name == phase_name]

    def phases_by_target_object(self, target_object: str) -> list[PhaseClaim]:
        """Return motion-phase claims whose target object equals *target_object*."""

        return [
            phase for phase in self.phases() if phase.target_object == target_object
        ]

    def phases_for_plan(self, plan: PlanClaim) -> list[PhaseClaim]:
        """Return the phase claims attached directly to *plan*."""

        return [
            phase
            for phase in plan.phases
            if phase.meta.source_reasoner == self.REASONER_NAME
        ]

    def contact_phases(self) -> list[PhaseClaim]:
        """Return motion phases that expect contact."""

        return [phase for phase in self.phases() if phase.contact]

    def preconditions(self) -> list[PreconditionClaim]:
        """Return Flanagan precondition claims preserving insertion order."""

        return [
            precondition
            for precondition in self.graph.domain(PreconditionClaim)
            if precondition.meta.source_reasoner == self.REASONER_NAME
        ]

    def goal_conditions(self) -> list[GoalConditionClaim]:
        """Return Flanagan goal-condition claims preserving insertion order."""

        return [
            goal_condition
            for goal_condition in self.graph.domain(GoalConditionClaim)
            if goal_condition.meta.source_reasoner == self.REASONER_NAME
        ]

    def failure_modes(self) -> list[FailureModeClaim]:
        """Return Flanagan failure-mode claims preserving insertion order."""

        return [
            failure_mode
            for failure_mode in self.graph.domain(FailureModeClaim)
            if failure_mode.meta.source_reasoner == self.REASONER_NAME
        ]

    def recovery_strategies(self) -> list[RecoveryStrategyClaim]:
        """Return Flanagan recovery-strategy claims preserving insertion order."""

        return [
            recovery_strategy
            for recovery_strategy in self.graph.domain(RecoveryStrategyClaim)
            if recovery_strategy.meta.source_reasoner == self.REASONER_NAME
        ]

    def preconditions_for_phase(self, phase: PhaseClaim) -> list[PreconditionClaim]:
        """Return preconditions attached directly to *phase*."""

        return [
            precondition
            for precondition in phase.preconditions
            if precondition.meta.source_reasoner == self.REASONER_NAME
        ]

    def goal_conditions_for_phase(
        self, phase: PhaseClaim
    ) -> list[GoalConditionClaim]:
        """Return goal conditions attached directly to *phase*."""

        return [
            goal_condition
            for goal_condition in phase.goal_conditions
            if goal_condition.meta.source_reasoner == self.REASONER_NAME
        ]

    def failure_modes_for_phase(self, phase: PhaseClaim) -> list[FailureModeClaim]:
        """Return failure modes attached directly to *phase*."""

        return [
            failure_mode
            for failure_mode in phase.failure_modes
            if failure_mode.meta.source_reasoner == self.REASONER_NAME
        ]

    def recovery_strategies_for_phase(
        self, phase: PhaseClaim
    ) -> list[RecoveryStrategyClaim]:
        """Return recovery strategies attached directly to *phase*."""

        return [
            recovery_strategy
            for recovery_strategy in phase.recovery_strategies
            if recovery_strategy.meta.source_reasoner == self.REASONER_NAME
        ]

    def phases_with_failures(self) -> list[PhaseClaim]:
        """Return motion phases that include explicit failure modes."""

        return [phase for phase in self.phases() if phase.failure_modes]

    def high_urgency_phases(self) -> list[PhaseClaim]:
        """Return motion phases marked as high urgency."""

        return [phase for phase in self.phases() if phase.urgency == "high"]

    def force_dynamic_events(self) -> list[ForceDynamicEvent]:
        """Return all force-dynamic boundary events from this reasoner."""

        return [
            e
            for e in self.graph.domain(ForceDynamicEvent)
            if e.meta.source_reasoner == self.REASONER_NAME
        ]

    def entry_event_for_phase(self, phase: PhaseClaim) -> ForceDynamicEvent | None:
        """Return the entry boundary event for *phase*, or None."""

        return phase.entry_event

    def exit_event_for_phase(self, phase: PhaseClaim) -> ForceDynamicEvent | None:
        """Return the exit boundary event for *phase*, or None."""

        return phase.exit_event
