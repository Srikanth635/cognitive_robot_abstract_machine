"""FrameNet-specific graph query facade."""

from __future__ import annotations

from dataclasses import dataclass

from llmr.hypotheses.elements import ClaimStatus, GroundingState
from llmr.hypotheses.projectors.framenet.constants import FRAMENET_REASONER_NAME
from llmr.hypotheses.projectors.framenet.edges import HasRoleEdge
from llmr.hypotheses.projectors.framenet.nodes import (
    FrameHypothesisNode,
    FrameRoleHypothesisNode,
)
from llmr.hypotheses.views.base import ReasonerGraphView


@dataclass(frozen=True)
class FrameNetGraphView(ReasonerGraphView):
    """Typed query facade for FrameNet hypothesis nodes and relations."""

    REASONER_NAME = FRAMENET_REASONER_NAME
    ROOT_CLAIM_TYPES = (FrameHypothesisNode,)
    CLAIM_TYPES = (FrameHypothesisNode, FrameRoleHypothesisNode)

    def frames(self) -> list[FrameHypothesisNode]:
        """Return FrameNet frame claims preserving graph insertion order."""

        return [
            node
            for node in self.graph.iter_nodes()
            if isinstance(node, FrameHypothesisNode)
            and node.meta.source_reasoner == self.REASONER_NAME
        ]

    def roles(self) -> list[FrameRoleHypothesisNode]:
        """Return FrameNet role claims preserving graph insertion order."""

        return [
            node
            for node in self.graph.iter_nodes()
            if isinstance(node, FrameRoleHypothesisNode)
            and node.meta.source_reasoner == self.REASONER_NAME
        ]

    def frames_by_frame(self, frame_name: str) -> list[FrameHypothesisNode]:
        """Return FrameNet frame claims whose frame label equals *frame_name*."""

        return [frame for frame in self.frames() if frame.frame == frame_name]

    def roles_by_role_name(self, role_name: str) -> list[FrameRoleHypothesisNode]:
        """Return FrameNet role claims whose role name equals *role_name*."""

        return [role for role in self.roles() if role.role_name == role_name]

    def roles_for_frame(
        self, frame_node: FrameHypothesisNode
    ) -> list[FrameRoleHypothesisNode]:
        """Return role claims attached to *frame_node* through HasRoleEdge."""

        return [
            role
            for role in self.graph.get_targets(
                frame_node.id, HasRoleEdge, FrameRoleHypothesisNode
            )
            if role.meta.source_reasoner == self.REASONER_NAME
        ]

    def grounded_roles(self) -> list[FrameRoleHypothesisNode]:
        """Return FrameNet roles grounded to structured world state."""

        return [
            role
            for role in self.roles()
            if role.meta.grounding == GroundingState.SYMBOL_GROUNDED
        ]

    def supported_roles(self) -> list[FrameRoleHypothesisNode]:
        """Return FrameNet roles with supported claim status."""

        return [
            role for role in self.roles() if role.meta.status == ClaimStatus.SUPPORTED
        ]

    def hypothesis_only_roles(self) -> list[FrameRoleHypothesisNode]:
        """Return FrameNet roles that remain pure hypotheses."""

        return [
            role
            for role in self.roles()
            if role.meta.status == ClaimStatus.HYPOTHESIS
            and role.meta.grounding == GroundingState.TEXT_ONLY
        ]
