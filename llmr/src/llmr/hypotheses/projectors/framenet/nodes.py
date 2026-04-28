"""FrameNet-specific hypothesis nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Optional

from llmr.hypotheses.common.nodes import ReasonerClaimNode


@dataclass
class FrameHypothesisNode(ReasonerClaimNode):
    """Frame-level hypothesis derived from a FrameNet interpretation."""

    frame: str
    lexical_unit: str
    framenet_label: str
    action_type: str
    instruction_text: Optional[str]


@dataclass
class FrameRoleHypothesisNode(ReasonerClaimNode):
    """One FrameNet role claim, flattened for graph querying."""

    role_family: str
    role_name: str
    filler_text: str
    filler_kind: str
    canonical_text: Optional[str] = None
