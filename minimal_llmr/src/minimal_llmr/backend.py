"""LLMBackend — KRROOD GenerativeBackend facade over ActionResolutionPipeline.

A single pipeline instance is created in __post_init__ and reused across all
_evaluate() calls, accumulating last_result for post-hoc inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.utils import T

from minimal_llmr.core.pipeline import ActionResolutionPipeline, ActionResolutionResult

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass
class LLMBackend(GenerativeBackend):
    """Thin KRROOD backend that delegates resolution to ActionResolutionPipeline."""

    llm: "BaseChatModel"
    instruction: Optional[str] = field(kw_only=True, default=None)
    world_context_provider: Optional[Callable[[], str]] = field(kw_only=True, default=None)
    strict_required: bool = field(kw_only=True, default=False)
    last_result: Optional[ActionResolutionResult] = field(default=None, init=False, repr=False)
    _pipeline: ActionResolutionPipeline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pipeline = ActionResolutionPipeline(
            llm=self.llm,
            instruction=self.instruction,
            strict_required=self.strict_required,
            world_context_provider=self.world_context_provider,
        )

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        result = self._pipeline.resolve(expression)
        self.last_result = result
        yield result.action
