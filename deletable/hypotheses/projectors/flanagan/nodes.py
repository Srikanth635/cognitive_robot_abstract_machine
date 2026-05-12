"""Flanagan-specific hypothesis nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from typing_extensions import Any, Dict, Optional, Tuple

from llmr.hypotheses.common.nodes import ProjectedClaimNode
from llmr.hypotheses.linked import GraphLinked
from llmr.hypotheses.projectors.flanagan.edges import HasPhaseEdge


@dataclass
class PlanClaimNode(ProjectedClaimNode, GraphLinked):
    """Top-level claim representing one Flanagan motion-phase plan."""

    action_type: str
    instruction_text: Optional[str]
    phase_count: int

    @property
    def phases(self) -> List[PhaseClaimNode]:
        return self.linked(HasPhaseEdge, PhaseClaimNode)

    @property
    def action(self):
        from llmr.hypotheses.common.nodes import ActionNode
        from llmr.hypotheses.common.edges import AboutActionEdge
        targets = self.linked(AboutActionEdge, ActionNode)
        return targets[0] if targets else None

    @property
    def dot_label(self) -> str:
        return f"Plan\\n{self.action_type} / {self.phase_count} phases"


@dataclass
class PhaseClaimNode(ProjectedClaimNode, GraphLinked):
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

    @property
    def plan(self) -> "PlanClaimNode":
        sources = self.linked_sources(HasPhaseEdge, PlanClaimNode)
        return sources[0] if sources else None

    @property
    def dot_label(self) -> str:
        return f"[{self.phase_index}] {self.phase_name}\\n-> {self.target_object}"
