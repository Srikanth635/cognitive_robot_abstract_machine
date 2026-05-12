"""Entrypoints for already-built KRROOD ``Match`` objects."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Optional

from krrood.symbol_graph.symbol_graph import Symbol

from llmr_updated_arch.core.backend import LLMBackend

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr_updated_arch.generation import Reasoner
    from llmr_updated_arch.hypotheses import BuildOrchestrator


def instance_from_match(
    match: Any,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
    reasoners: Optional[list["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> Any:
    """Resolve an underspecified ``Match`` and return the concrete action instance."""

    backend = make_backend_for_match(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return next(iter(backend.evaluate(match)))


def make_backend_for_match(
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
    reasoners: Optional[list["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> LLMBackend:
    """Create the backend used by match-oriented entrypoints."""

    return LLMBackend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners or [],
        sg_model_orchestrator=sg_model_orchestrator,
    )
