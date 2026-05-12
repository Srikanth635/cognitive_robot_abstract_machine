"""Epistemic hypothesis graph for llmr reasoner outputs.

This package stores LLM-generated interpretations as typed graph nodes and
edges. The graph is intentionally separate from KRROOD's SymbolGraph:

- SymbolGraph models world entities and relations.
- HypothesisGraph models reasoner-produced claims, their structure, and their
  grounding/support metadata.
"""

from llmr.hypotheses.elements import (
    ClaimStatus,
    GroundingState,
    HypothesisEdge,
    HypothesisGraphElement,
    HypothesisMeta,
    HypothesisNode,
)
from llmr.hypotheses.families import HypothesisFamily
from llmr.hypotheses.common.edges import (
    AboutActionEdge,
    GroundedByEdge,
    ProducesClaimEdge,
    SupportedByEdge,
)
from llmr.hypotheses.projectors.flanagan.edges import (
    EvokesPlanEdge,
    HasPhaseEdge,
)
from llmr.hypotheses.projectors.framenet.edges import EvokesFrameEdge, HasRoleEdge
from llmr.hypotheses.graph import HypothesisGraph
from llmr.hypotheses.projection import (
    ProjectionOrchestrator,
    HypothesisProjection,
    ProjectionInput,
    HypothesisProjector,
    ProjectorRegistry,
)
from llmr.hypotheses.projectors.framenet import (
    FrameNetFamily,
    FrameNetGraphView,
    FrameNetProjector,
)
from llmr.hypotheses.projectors.flanagan import (
    FlanaganFamily,
    FlanaganGraphView,
    FlanaganProjector,
)
from llmr.hypotheses.views import HypothesisGraphView, ReasonerGraphView
from llmr.hypotheses.common.nodes import (
    ActionNode,
    ClaimNode,
    AnchorNode,
    EvidenceNode,
    InstructionNode,
    ProjectedClaimNode,
    ReasonerRunNode,
    SlotEvidenceNode,
    GroundingEvidenceNode,
)
from llmr.hypotheses.projectors.framenet.nodes import (
    FrameClaimNode,
    RoleClaimNode,
)
from llmr.hypotheses.projectors.flanagan.nodes import (
    PhaseClaimNode,
    PlanClaimNode,
)
from llmr.hypotheses.algorithms import (
    conflicting_role_claims,
    hypothesis_closure,
    invalidate_from_symbol,
    reasoning_chain,
)
from llmr.hypotheses.linked import GraphLinked
from llmr.hypotheses.families.base import hypothesis_family, get_all_families

__all__ = [
    "ClaimStatus",
    "GroundingState",
    "HypothesisMeta",
    "HypothesisGraphElement",
    "HypothesisNode",
    "HypothesisEdge",
    "HypothesisFamily",
    "AnchorNode",
    "ClaimNode",
    "ProjectedClaimNode",
    "EvidenceNode",
    "InstructionNode",
    "ActionNode",
    "ReasonerRunNode",
    "PlanClaimNode",
    "PhaseClaimNode",
    "FrameClaimNode",
    "RoleClaimNode",
    "SlotEvidenceNode",
    "GroundingEvidenceNode",
    "ProducesClaimEdge",
    "EvokesPlanEdge",
    "EvokesFrameEdge",
    "AboutActionEdge",
    "HasPhaseEdge",
    "HasRoleEdge",
    "SupportedByEdge",
    "GroundedByEdge",
    "HypothesisGraph",
    "HypothesisGraphView",
    "ReasonerGraphView",
    "ProjectionInput",
    "ProjectionOrchestrator",
    "ProjectorRegistry",
    "HypothesisProjection",
    "HypothesisProjector",
    "FlanaganFamily",
    "FlanaganProjector",
    "FlanaganGraphView",
    "FrameNetFamily",
    "FrameNetProjector",
    "FrameNetGraphView",
    "invalidate_from_symbol",
    "reasoning_chain",
    "conflicting_role_claims",
    "hypothesis_closure",
    "GraphLinked",
    "hypothesis_family",
    "get_all_families",
]
