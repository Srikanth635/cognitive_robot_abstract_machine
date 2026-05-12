"""Common anchor and evidence entities for the sg_model object model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from typing_extensions import Any, Optional

from llmr_updated_arch.hypotheses.entities.base import (
    AnchorHypothesis,
    ClaimHypothesis,
    EvidenceHypothesis,
)

if TYPE_CHECKING:
    from llmr_updated_arch.hypotheses.entities.flanagan import PlanClaim
    from llmr_updated_arch.hypotheses.entities.framenet import FrameClaim


@dataclass(eq=False)
class Instruction(AnchorHypothesis):
    """Normalized instruction anchor for sg_model queries."""

    text: str
    normalized_text: str
    frames: list["FrameClaim"] = field(
        default_factory=list, repr=False, compare=False
    )
    plans: list["PlanClaim"] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_frame(self, frame: "FrameClaim") -> "FrameClaim":
        """Attach *frame* to this instruction, maintaining the inverse link."""

        previous_instruction = getattr(frame, "instruction", None)
        if previous_instruction is not None and previous_instruction is not self:
            try:
                previous_instruction.frames.remove(frame)
            except ValueError:
                pass
        if frame not in self.frames:
            self.frames.append(frame)
        frame.instruction = self
        return frame

    def add_plan(self, plan: "PlanClaim") -> "PlanClaim":
        """Attach *plan* to this instruction, maintaining the inverse link."""

        previous_instruction = getattr(plan, "instruction", None)
        if previous_instruction is not None and previous_instruction is not self:
            try:
                previous_instruction.plans.remove(plan)
            except ValueError:
                pass
        if plan not in self.plans:
            self.plans.append(plan)
        plan.instruction = self
        return plan


@dataclass(eq=False)
class Action(AnchorHypothesis):
    """Anchor for a resolved concrete action instance."""

    action_ref: Any
    action_type: str
    frame_claims: list["FrameClaim"] = field(
        default_factory=list, repr=False, compare=False
    )
    plan_claims: list["PlanClaim"] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_frame_claim(self, frame: "FrameClaim") -> "FrameClaim":
        """Attach *frame* to this action, maintaining the inverse link."""

        previous_action = getattr(frame, "action", None)
        if previous_action is not None and previous_action is not self:
            try:
                previous_action.frame_claims.remove(frame)
            except ValueError:
                pass
        if frame not in self.frame_claims:
            self.frame_claims.append(frame)
        frame.action = self
        return frame

    def add_plan_claim(self, plan: "PlanClaim") -> "PlanClaim":
        """Attach *plan* to this action, maintaining the inverse link."""

        previous_action = getattr(plan, "action", None)
        if previous_action is not None and previous_action is not self:
            try:
                previous_action.plan_claims.remove(plan)
            except ValueError:
                pass
        if plan not in self.plan_claims:
            self.plan_claims.append(plan)
        plan.action = self
        return plan


@dataclass(eq=False)
class ReasonerRun(AnchorHypothesis):
    """Represents one reasoner invocation associated with an action."""

    reasoner_name: str
    run_id: str
    model_name: Optional[str]
    prompt_version: Optional[str]
    action_type: str
    instruction_text: Optional[str]
    claims: list[ClaimHypothesis] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_claim(self, claim: ClaimHypothesis) -> ClaimHypothesis:
        """Attach *claim* to this run, maintaining the inverse link when present."""

        previous_run = getattr(claim, "run", None)
        if previous_run is not None and previous_run is not self:
            try:
                previous_run.claims.remove(claim)
            except ValueError:
                pass
        if claim not in self.claims:
            self.claims.append(claim)
        if hasattr(claim, "run"):
            claim.run = self
        return claim


@dataclass(eq=False)
class SlotEvidence(EvidenceHypothesis):
    """Evidence that a claim aligns with a resolved action slot."""

    slot_name: str
    value_ref: Any
    value_repr: str


@dataclass(eq=False)
class GroundingEvidence(EvidenceHypothesis):
    """Evidence that a claim was grounded to a structured world object."""

    query_text: str
    symbol_ref: Any
    symbol_type: str
    grounding_method: str
    ambiguity_note: Optional[str] = None
