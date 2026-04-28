"""Flanagan-specific hypothesis nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any, Dict, Optional, Tuple

from llmr.hypotheses.common.nodes import ReasonerClaimNode


@dataclass
class MotionPlanHypothesisNode(ReasonerClaimNode):
    """Top-level claim representing one Flanagan motion-phase plan."""

    action_type: str
    instruction_text: Optional[str]
    phase_count: int


@dataclass
class MotionPhaseHypothesisNode(ReasonerClaimNode):
    """One motion phase from a Flanagan motion-phase plan."""

    phase_index: int
    phase_name: str
    target_object: str
    description: Optional[str]
    symbol: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    goal_state: Dict[str, Any] = field(default_factory=dict)
    force_dynamics: Dict[str, Any] = field(default_factory=dict)
    sensory_feedback: Dict[str, Any] = field(default_factory=dict)
    failure_and_recovery: Dict[str, Any] = field(default_factory=dict)
    temporal_constraints: Dict[str, Any] = field(default_factory=dict)
    contact: bool = False
    motion_type: Optional[str] = None
    max_duration_sec: Optional[float] = None
    urgency: Optional[str] = None
    possible_failures: Tuple[str, ...] = ()
    recovery_strategies: Tuple[str, ...] = ()
