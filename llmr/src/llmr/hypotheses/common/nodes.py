"""Reusable hypothesis graph nodes shared across reasoners."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing_extensions import Any, Optional

from llmr.hypotheses.elements import HypothesisNode

if False:  # pragma: no cover
    from krrood.symbol_graph.symbol_graph import Symbol


@dataclass
class ContextNode(HypothesisNode, ABC):
    """Abstract node anchoring claims to an instruction, action, or run."""


@dataclass
class ClaimNode(HypothesisNode, ABC):
    """Abstract node representing an epistemic claim."""


@dataclass
class ReasonerClaimNode(ClaimNode, ABC):
    """Claim node produced by a pluggable llmr reasoner."""


@dataclass
class EvidenceNode(HypothesisNode, ABC):
    """Abstract node providing structured support for a claim."""


@dataclass
class InstructionNode(ContextNode):
    """Normalized instruction anchor for hypothesis graph queries."""

    text: str
    normalized_text: str


@dataclass
class ActionNode(ContextNode):
    """Anchor node for a resolved concrete action instance."""

    action_ref: Any
    action_type: str


@dataclass
class ReasonerRunNode(ContextNode):
    """Represents one reasoner invocation associated with an action."""

    reasoner_name: str
    run_id: str
    model_name: Optional[str]
    prompt_version: Optional[str]
    action_type: str
    instruction_text: Optional[str]


@dataclass
class SlotBindingEvidenceNode(EvidenceNode):
    """Evidence that a claim aligns with a resolved action slot."""

    slot_name: str
    value_ref: Any
    value_repr: str


@dataclass
class SymbolGroundingEvidenceNode(EvidenceNode):
    """Evidence that a claim was grounded to a concrete Symbol instance."""

    query_text: str
    symbol_ref: "Symbol"
    symbol_type: str
    grounding_method: str
    ambiguity_note: Optional[str] = None
