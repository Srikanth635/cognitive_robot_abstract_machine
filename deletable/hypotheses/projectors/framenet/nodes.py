"""FrameNet-specific hypothesis nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from typing_extensions import Optional

from llmr.hypotheses.common.nodes import ProjectedClaimNode
from llmr.hypotheses.linked import GraphLinked
from llmr.hypotheses.projectors.framenet.edges import HasRoleEdge


@dataclass
class FrameClaimNode(ProjectedClaimNode, GraphLinked):
    """Frame-level claim derived from a FrameNet interpretation."""

    frame: str
    lexical_unit: str
    framenet_label: str
    action_type: str
    instruction_text: Optional[str]

    @property
    def roles(self) -> List[RoleClaimNode]:
        return self.linked(HasRoleEdge, RoleClaimNode)

    @property
    def action(self):
        from llmr.hypotheses.common.nodes import ActionNode
        from llmr.hypotheses.common.edges import AboutActionEdge
        targets = self.linked(AboutActionEdge, ActionNode)
        return targets[0] if targets else None

    @property
    def dot_label(self) -> str:
        return f"Frame: {self.frame}\\n{self.lexical_unit}"


@dataclass
class RoleClaimNode(ProjectedClaimNode, GraphLinked):
    """One FrameNet role claim, flattened for graph querying."""

    role_family: str
    role_name: str
    filler_text: str
    filler_kind: str
    canonical_text: Optional[str] = None

    @property
    def frame(self) -> "FrameClaimNode":
        sources = self.linked_sources(HasRoleEdge, FrameClaimNode)
        return sources[0] if sources else None

    @property
    def dot_label(self) -> str:
        return f"{self.role_name}: {self.filler_text}\\n({self.filler_kind})"
