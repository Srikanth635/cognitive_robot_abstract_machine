"""FrameNet-specific view facade over sg_model objects."""

from __future__ import annotations

from dataclasses import dataclass

from llmr.hypotheses.builders.framenet import FRAMENET_REASONER_NAME
from llmr.hypotheses.entities.framenet import FrameClaim, RoleClaim
from llmr.hypotheses.meta import ClaimStatus, GroundingState
from llmr.hypotheses.views.base import ReasonerGraphView


@dataclass(frozen=True)
class FrameNetGraphView(ReasonerGraphView):
    """Typed query facade for FrameNet hypothesis objects."""

    REASONER_NAME = FRAMENET_REASONER_NAME
    ROOT_CLAIM_TYPES = (FrameClaim,)
    CLAIM_TYPES = (FrameClaim, RoleClaim)

    def frames(self) -> list[FrameClaim]:
        """Return FrameNet frame claims preserving repository insertion order."""

        return [
            frame
            for frame in self.graph.domain(FrameClaim)
            if frame.meta.source_reasoner == self.REASONER_NAME
        ]

    def roles(self) -> list[RoleClaim]:
        """Return FrameNet role claims preserving repository insertion order."""

        return [
            role
            for role in self.graph.domain(RoleClaim)
            if role.meta.source_reasoner == self.REASONER_NAME
        ]

    def frames_by_frame(self, frame_name: str) -> list[FrameClaim]:
        """Return FrameNet frame claims whose frame label equals *frame_name*."""

        return [frame for frame in self.frames() if frame.frame == frame_name]

    def roles_by_role_name(self, role_name: str) -> list[RoleClaim]:
        """Return FrameNet role claims whose role name equals *role_name*."""

        return [role for role in self.roles() if role.role_name == role_name]

    def roles_for_frame(self, frame: FrameClaim) -> list[RoleClaim]:
        """Return the role claims attached directly to *frame*."""

        return [
            role
            for role in frame.roles
            if role.meta.source_reasoner == self.REASONER_NAME
        ]

    def grounded_roles(self) -> list[RoleClaim]:
        """Return FrameNet roles grounded to structured world state."""

        return [
            role
            for role in self.roles()
            if role.meta.grounding == GroundingState.SYMBOL_GROUNDED
        ]

    def supported_roles(self) -> list[RoleClaim]:
        """Return FrameNet roles with supported claim status."""

        return [
            role
            for role in self.roles()
            if role.meta.status == ClaimStatus.SUPPORTED
        ]

    def hypothesis_only_roles(self) -> list[RoleClaim]:
        """Return FrameNet roles that remain pure hypotheses."""

        return [
            role
            for role in self.roles()
            if role.meta.status == ClaimStatus.HYPOTHESIS
            and role.meta.grounding == GroundingState.TEXT_ONLY
        ]
