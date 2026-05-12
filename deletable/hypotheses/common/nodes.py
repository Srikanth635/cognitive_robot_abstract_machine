"""Reusable hypothesis graph nodes shared across reasoners."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing_extensions import Any, Optional

from llmr.hypotheses.elements import HypothesisNode
from llmr.hypotheses.linked import GraphLinked

if False:  # pragma: no cover
    from krrood.symbol_graph.symbol_graph import Symbol


@dataclass
class AnchorNode(HypothesisNode, ABC):
    """Abstract node anchoring claims to an instruction, action, or run."""


@dataclass
class ClaimNode(HypothesisNode, ABC):
    """Abstract node representing an epistemic claim."""


@dataclass
class ProjectedClaimNode(ClaimNode, ABC):
    """Claim node produced by a pluggable llmr reasoner via the projection pipeline."""


@dataclass
class EvidenceNode(HypothesisNode, ABC):
    """Abstract node providing structured support for a claim."""


@dataclass
class InstructionNode(AnchorNode, GraphLinked):
    """Normalized instruction anchor for hypothesis graph queries."""

    text: str
    normalized_text: str

    @property
    def frames(self):
        """Return FrameClaimNodes evoked by this instruction."""
        from llmr.hypotheses.projectors.framenet.nodes import FrameClaimNode
        from llmr.hypotheses.projectors.framenet.edges import EvokesFrameEdge
        return self.linked(EvokesFrameEdge, FrameClaimNode)

    @property
    def plans(self):
        """Return PlanClaimNodes evoked by this instruction."""
        from llmr.hypotheses.projectors.flanagan.nodes import PlanClaimNode
        from llmr.hypotheses.projectors.flanagan.edges import EvokesPlanEdge
        return self.linked(EvokesPlanEdge, PlanClaimNode)

    @property
    def dot_label(self) -> str:
        snippet = self.text[:40] + ("..." if len(self.text) > 40 else "")
        return f"Instruction\\n{snippet}"


@dataclass
class ActionNode(AnchorNode, GraphLinked):
    """Anchor node for a resolved concrete action instance."""

    action_ref: Any
    action_type: str

    @property
    def frames(self):
        """Return FrameClaimNodes whose AboutActionEdge points to this action."""
        from llmr.hypotheses.projectors.framenet.nodes import FrameClaimNode
        from llmr.hypotheses.common.edges import AboutActionEdge
        return self.linked_sources(AboutActionEdge, FrameClaimNode)

    @property
    def plans(self):
        """Return PlanClaimNodes whose AboutActionEdge points to this action."""
        from llmr.hypotheses.projectors.flanagan.nodes import PlanClaimNode
        from llmr.hypotheses.common.edges import AboutActionEdge
        return self.linked_sources(AboutActionEdge, PlanClaimNode)

    @property
    def dot_label(self) -> str:
        return f"Action\\n{self.action_type}"


@dataclass
class ReasonerRunNode(AnchorNode):
    """Represents one reasoner invocation associated with an action."""

    reasoner_name: str
    run_id: str
    model_name: Optional[str]
    prompt_version: Optional[str]
    action_type: str
    instruction_text: Optional[str]

    @property
    def dot_label(self) -> str:
        return f"Run [{self.run_id[:8]}]\\n{self.reasoner_name}"


@dataclass
class SlotEvidenceNode(EvidenceNode):
    """Evidence that a claim aligns with a resolved action slot."""

    slot_name: str
    value_ref: Any
    value_repr: str

    @property
    def dot_label(self) -> str:
        return f"Slot: {self.slot_name}\\n{self.value_repr}"


@dataclass
class GroundingEvidenceNode(EvidenceNode):
    """Evidence that a claim was grounded to a concrete Symbol instance."""

    query_text: str
    symbol_ref: "Symbol"
    symbol_type: str
    grounding_method: str
    ambiguity_note: Optional[str] = None

    @property
    def dot_label(self) -> str:
        return f"Grounded: {self.query_text}\\n{self.symbol_type}"
