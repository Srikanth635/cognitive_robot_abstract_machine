"""KRROOD ``GenerativeBackend`` facade over ``ActionResolutionPipeline``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Iterable, List, Optional, Type

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.utils import T
from krrood.symbol_graph.symbol_graph import Symbol

from llmr_updated_arch.core.pipeline import ActionResolutionPipeline
from llmr_updated_arch.schemas import ActionClassificationResult, SemanticBundle

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr_updated_arch.generation import Reasoner
    from llmr_updated_arch.hypotheses import BuildOrchestrator, BuildResult, HypothesisGraph


@dataclass
class LLMBackend(GenerativeBackend):
    """Thin KRROOD backend facade that delegates resolution to the pipeline."""

    llm: "BaseChatModel"
    symbol_type: Type[Symbol] = field(default=Symbol)
    instruction: Optional[str] = field(kw_only=True, default=None)
    world_context_provider: Optional[Callable[[], str]] = field(kw_only=True, default=None)
    strict_required: bool = field(kw_only=True, default=False)
    reasoners: List["Reasoner"] = field(default_factory=list, kw_only=True)
    classification: Optional[ActionClassificationResult] = field(default=None, kw_only=True)
    sg_model_orchestrator: Optional["BuildOrchestrator"] = field(
        default=None, kw_only=True, repr=False
    )

    semantics: Optional[SemanticBundle] = field(default=None, init=False, repr=False)
    last_result: Optional[Any] = field(default=None, init=False, repr=False)
    last_build_result: Optional["BuildResult"] = field(default=None, init=False, repr=False)

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        pipeline = self._make_pipeline()
        result = pipeline.resolve(expression)
        self.sg_model_orchestrator = pipeline.sg_model_orchestrator
        self.semantics = result.semantic_bundle
        self.last_result = result
        self.last_build_result = result.projection_result
        yield result.action

    @property
    def hypothesis_graph(self) -> Optional["HypothesisGraph"]:
        if self.sg_model_orchestrator is None:
            return None
        return self.sg_model_orchestrator.graph

    def _make_pipeline(self) -> ActionResolutionPipeline:
        return ActionResolutionPipeline(
            llm=self.llm,
            symbol_type=self.symbol_type,
            instruction=self.instruction,
            world_context_provider=self.world_context_provider,
            strict_required=self.strict_required,
            classification=self.classification,
            reasoners=list(self.reasoners),
            sg_model_orchestrator=self.sg_model_orchestrator,
        )
