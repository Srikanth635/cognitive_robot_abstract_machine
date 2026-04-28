"""Flanagan-specific hypothesis nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from typing_extensions import Any, Dict, Optional, Tuple

from llmr.hypotheses.common.nodes import ReasonerClaimNode
from llmr.hypotheses.linked import GraphLinked
from llmr.hypotheses.projectors.flanagan.edges import HasMotionPhaseEdge


@dataclass
class MotionPlanHypothesisNode(ReasonerClaimNode, GraphLinked):
    """Top-level claim representing one Flanagan motion-phase plan."""

    action_type: str
    instruction_text: Optional[str]
    phase_count: int

    @property
    def phases(self) -> List[MotionPhaseHypothesisNode]:
        """Return phase nodes attached to this plan via HasMotionPhaseEdge."""
        return self.linked(HasMotionPhaseEdge, MotionPhaseHypothesisNode)


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
