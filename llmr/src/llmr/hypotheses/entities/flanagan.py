"""Flanagan-specific claim entities for the sg_model object model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from typing_extensions import Any

from llmr.hypotheses.entities.base import ProjectedClaimHypothesis
from llmr.hypotheses.entities.common import Action, Instruction, ReasonerRun


@dataclass
class TemporalInterval:
    """Hypothetical time bounds for a motion phase.

    Derived from the LLM-estimated ``max_duration_sec`` and the cumulative
    offset of preceding phases.  ``is_hypothetical=True`` until grounded to
    an actual execution episode.
    """

    offset_from_start_sec: float
    duration_sec: float
    is_hypothetical: bool = True

    @property
    def end_sec(self) -> float:
        return self.offset_from_start_sec + self.duration_sec


@dataclass(eq=False)
class ForceDynamicEvent(ProjectedClaimHypothesis):
    """A force-dynamic event that marks the entry or exit boundary of a phase.

    Derived from the canonical phase vocabulary — not from live physics.
    ``is_hypothetical`` on the linked ``TemporalInterval`` signals this.
    """

    event_type: str
    """Canonical event name, e.g. ``EffectorTouchesObject``, ``ObjectLiftOff``."""

    agent: Optional[str]
    """Effector or body part involved, as a text reference."""

    object_ref: str
    """Target object involved, as a text reference."""

    expected_offset_sec: Optional[float] = None
    """Estimated time from action start when this event occurs."""

    role: str = "entry"
    """``"entry"`` if this event opens the phase, ``"exit"`` if it closes it."""

    phase: Optional["PhaseClaim"] = field(default=None, repr=False, compare=False)


@dataclass(eq=False)
class PlanClaim(ProjectedClaimHypothesis):
    """Top-level claim representing one Flanagan motion-phase plan."""

    action_type: str
    instruction_text: Optional[str]
    phase_count: int
    instruction: Optional[Instruction] = field(
        default=None, repr=False, compare=False
    )
    action: Optional[Action] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)
    phases: list["PhaseClaim"] = field(default_factory=list, repr=False, compare=False)

    def add_phase(self, phase: "PhaseClaim") -> "PhaseClaim":
        """Attach *phase* to this plan, maintaining the inverse link."""

        previous_plan = phase.plan
        if previous_plan is not None and previous_plan is not self:
            try:
                previous_plan.phases.remove(phase)
            except ValueError:
                pass
        if phase not in self.phases:
            self.phases.append(phase)
        phase.plan = self
        return phase


@dataclass(eq=False)
class PreconditionClaim(ProjectedClaimHypothesis):
    """One evaluable precondition that must hold before a Flanagan phase begins."""

    predicate_name: str
    """Canonical predicate name, e.g. ``is_near_by``, ``gripper_open``."""

    subject: Optional[str] = None
    """Primary entity involved (text ref). Bound to a world object at grounding time."""

    object_ref: Optional[str] = None
    """Secondary entity for relational predicates (text ref). Defaulted from phase target_object at build time."""

    expected_value: bool = True
    """True if the predicate must hold; False if it must not hold."""

    predicate_ref: Optional[str] = None
    """Name of the callable in predicates.py that evaluates this condition at runtime."""

    source_key: Optional[str] = None
    phase: Optional["PhaseClaim"] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)


@dataclass(eq=False)
class GoalConditionClaim(ProjectedClaimHypothesis):
    """One evaluable goal condition that must hold after a Flanagan phase completes."""

    predicate_name: str
    """Canonical predicate name, e.g. ``is_pregrasp_aligned``, ``object_stable``."""

    subject: Optional[str] = None
    """Primary entity involved (text ref). Bound to a world object at grounding time."""

    object_ref: Optional[str] = None
    """Secondary entity for relational predicates (text ref). Defaulted from phase target_object at build time."""

    expected_value: bool = True
    """True if the predicate must hold; False if it must not hold."""

    predicate_ref: Optional[str] = None
    """Name of the callable in predicates.py that evaluates this condition at runtime."""

    source_key: Optional[str] = None
    phase: Optional["PhaseClaim"] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)


@dataclass(eq=False)
class FailureModeClaim(ProjectedClaimHypothesis):
    """One queryable failure mode attached to a Flanagan phase."""

    name: str
    value_text: Optional[str] = None
    source_key: Optional[str] = None
    phase: Optional["PhaseClaim"] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)


@dataclass(eq=False)
class RecoveryStrategyClaim(ProjectedClaimHypothesis):
    """One queryable recovery strategy attached to a Flanagan phase."""

    name: str
    value_text: Optional[str] = None
    source_key: Optional[str] = None
    phase: Optional["PhaseClaim"] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)


@dataclass(eq=False)
class PhaseClaim(ProjectedClaimHypothesis):
    """One motion phase from a Flanagan motion-phase plan."""

    phase_index: int
    phase_name: str
    target_object: str
    description: Optional[str]
    symbol: str
    force_dynamics: dict[str, Any] = field(default_factory=dict)
    sensory_feedback: dict[str, Any] = field(default_factory=dict)
    contact: bool = False
    motion_type: Optional[str] = None
    max_duration_sec: Optional[float] = None
    urgency: Optional[str] = None
    preconditions: list[PreconditionClaim] = field(
        default_factory=list, repr=False, compare=False
    )
    goal_conditions: list[GoalConditionClaim] = field(
        default_factory=list, repr=False, compare=False
    )
    failure_modes: list[FailureModeClaim] = field(
        default_factory=list, repr=False, compare=False
    )
    recovery_strategies: list[RecoveryStrategyClaim] = field(
        default_factory=list, repr=False, compare=False
    )
    plan: Optional[PlanClaim] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)
    temporal_interval: Optional[TemporalInterval] = field(
        default=None, repr=False, compare=False
    )
    entry_event: Optional[ForceDynamicEvent] = field(
        default=None, repr=False, compare=False
    )
    exit_event: Optional[ForceDynamicEvent] = field(
        default=None, repr=False, compare=False
    )

    def add_entry_event(self, event: ForceDynamicEvent) -> ForceDynamicEvent:
        """Attach *event* as the entry boundary of this phase."""
        self.entry_event = event
        event.phase = self
        event.role = "entry"
        return event

    def add_exit_event(self, event: ForceDynamicEvent) -> ForceDynamicEvent:
        """Attach *event* as the exit boundary of this phase."""
        self.exit_event = event
        event.phase = self
        event.role = "exit"
        return event

    def add_precondition(
        self, precondition: PreconditionClaim
    ) -> PreconditionClaim:
        """Attach *precondition* to this phase, maintaining the inverse link."""

        previous_phase = precondition.phase
        if previous_phase is not None and previous_phase is not self:
            try:
                previous_phase.preconditions.remove(precondition)
            except ValueError:
                pass
        if precondition not in self.preconditions:
            self.preconditions.append(precondition)
        precondition.phase = self
        return precondition

    def add_goal_condition(
        self, goal_condition: GoalConditionClaim
    ) -> GoalConditionClaim:
        """Attach *goal_condition* to this phase, maintaining the inverse link."""

        previous_phase = goal_condition.phase
        if previous_phase is not None and previous_phase is not self:
            try:
                previous_phase.goal_conditions.remove(goal_condition)
            except ValueError:
                pass
        if goal_condition not in self.goal_conditions:
            self.goal_conditions.append(goal_condition)
        goal_condition.phase = self
        return goal_condition

    def add_failure_mode(self, failure_mode: FailureModeClaim) -> FailureModeClaim:
        """Attach *failure_mode* to this phase, maintaining the inverse link."""

        previous_phase = failure_mode.phase
        if previous_phase is not None and previous_phase is not self:
            try:
                previous_phase.failure_modes.remove(failure_mode)
            except ValueError:
                pass
        if failure_mode not in self.failure_modes:
            self.failure_modes.append(failure_mode)
        failure_mode.phase = self
        return failure_mode

    def add_recovery_strategy(
        self, recovery_strategy: RecoveryStrategyClaim
    ) -> RecoveryStrategyClaim:
        """Attach *recovery_strategy* to this phase, maintaining the inverse link."""

        previous_phase = recovery_strategy.phase
        if previous_phase is not None and previous_phase is not self:
            try:
                previous_phase.recovery_strategies.remove(recovery_strategy)
            except ValueError:
                pass
        if recovery_strategy not in self.recovery_strategies:
            self.recovery_strategies.append(recovery_strategy)
        recovery_strategy.phase = self
        return recovery_strategy
