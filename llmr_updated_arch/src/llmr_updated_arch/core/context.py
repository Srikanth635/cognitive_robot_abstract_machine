"""Shared state passed through the resolution pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Optional

from krrood.symbol_graph.symbol_graph import Symbol

from llmr_updated_arch.integrations.krrood.match_reader import MatchSnapshot
from llmr_updated_arch.schemas import ActionClassificationResult, SemanticBundle

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr_updated_arch.generation import Reasoner
    from llmr_updated_arch.hypotheses import BuildOrchestrator


@dataclass
class ResolutionOptions:
    """Options that affect generation, grounding, validation, and projection."""

    strict_required: bool = False
    world_context_provider: Optional[Callable[[], str]] = None
    classification: Optional[ActionClassificationResult] = None
    reasoners: list["Reasoner"] = field(default_factory=list)
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None


@dataclass
class ResolutionContext:
    """Normalized input plus mutable semantic state for one action resolution."""

    instruction: Optional[str]
    match_snapshot: MatchSnapshot
    world_context: str
    llm: "BaseChatModel"
    symbol_type: type = Symbol
    semantic_bundle: SemanticBundle = field(default_factory=SemanticBundle)
    options: ResolutionOptions = field(default_factory=ResolutionOptions)
