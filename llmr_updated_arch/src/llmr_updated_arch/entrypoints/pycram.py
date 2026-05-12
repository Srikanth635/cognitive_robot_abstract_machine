"""PyCRAM plan entrypoints for match-driven execution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Optional

from krrood.symbol_graph.symbol_graph import Symbol

from llmr_updated_arch.entrypoints.match import make_backend_for_match

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr_updated_arch.generation import Reasoner
    from llmr_updated_arch.hypotheses import BuildOrchestrator
    from llmr_updated_arch.integrations.pycram import PycramContext, PycramPlanNode


def plan_from_match(
    match: Any,
    context: "PycramContext",
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
    reasoners: Optional[list["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> "PycramPlanNode":
    """Return a PyCRAM plan node for an already-built underspecified ``Match``."""

    from llmr_updated_arch.integrations.pycram import execute_single

    context.query_backend = make_backend_for_match(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return execute_single(match, context)
