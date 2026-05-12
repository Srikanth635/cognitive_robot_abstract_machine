"""FrameNet-specific claim entities for the sg_model object model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llmr.hypotheses.entities.base import ProjectedClaimHypothesis
from llmr.hypotheses.entities.common import (
    Action,
    GroundingEvidence,
    Instruction,
    ReasonerRun,
    SlotEvidence,
)


@dataclass(eq=False)
class FrameClaim(ProjectedClaimHypothesis):
    """Frame-level claim derived from a FrameNet interpretation."""

    frame: str
    lexical_unit: str
    framenet_label: str
    action_type: str
    instruction_text: Optional[str]
    instruction: Optional[Instruction] = field(
        default=None, repr=False, compare=False
    )
    action: Optional[Action] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)
    roles: list["RoleClaim"] = field(default_factory=list, repr=False, compare=False)

    def add_role(self, role: "RoleClaim") -> "RoleClaim":
        """Attach *role* to this frame, maintaining the inverse link."""

        previous_frame = role.frame
        if previous_frame is not None and previous_frame is not self:
            try:
                previous_frame.roles.remove(role)
            except ValueError:
                pass
        if role not in self.roles:
            self.roles.append(role)
        role.frame = self
        return role


@dataclass(eq=False)
class RoleClaim(ProjectedClaimHypothesis):
    """One FrameNet role claim associated with a frame claim."""

    role_family: str
    role_name: str
    filler_text: str
    filler_kind: str
    canonical_text: Optional[str] = None
    frame: Optional[FrameClaim] = field(default=None, repr=False, compare=False)
    run: Optional[ReasonerRun] = field(default=None, repr=False, compare=False)
    supported_by: list[SlotEvidence] = field(
        default_factory=list, repr=False, compare=False
    )
    grounded_by: list[GroundingEvidence] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_support(self, evidence: SlotEvidence) -> SlotEvidence:
        """Attach *evidence* as slot-alignment support for this role."""

        if evidence not in self.supported_by:
            self.supported_by.append(evidence)
        return evidence

    def add_grounding(self, evidence: GroundingEvidence) -> GroundingEvidence:
        """Attach *evidence* as grounding support for this role."""

        if evidence not in self.grounded_by:
            self.grounded_by.append(evidence)
        return evidence
