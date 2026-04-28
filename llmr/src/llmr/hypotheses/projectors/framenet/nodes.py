"""FrameNet-specific hypothesis nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from typing_extensions import Optional

from llmr.hypotheses.common.nodes import ReasonerClaimNode
from llmr.hypotheses.linked import GraphLinked
from llmr.hypotheses.projectors.framenet.edges import HasRoleEdge


@dataclass
class FrameHypothesisNode(ReasonerClaimNode, GraphLinked):
    """Frame-level hypothesis derived from a FrameNet interpretation."""

    frame: str
    lexical_unit: str
    framenet_label: str
    action_type: str
    instruction_text: Optional[str]

    @property
    def roles(self) -> List[FrameRoleHypothesisNode]:
        """Return role nodes attached to this frame via HasRoleEdge."""
        return self.linked(HasRoleEdge, FrameRoleHypothesisNode)


@dataclass
class FrameRoleHypothesisNode(ReasonerClaimNode):
    """One FrameNet role claim, flattened for graph querying."""

    role_family: str
    role_name: str
    filler_text: str
    filler_kind: str
    canonical_text: Optional[str] = None
