"""Natural-language instruction entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, Optional

from krrood.symbol_graph.symbol_graph import Symbol

from llmr_updated_arch.core.errors import LLMActionClassificationFailed
from llmr_updated_arch.entrypoints.match import instance_from_match
from llmr_updated_arch.generation.classifier import infer_action_class
from llmr_updated_arch.integrations.krrood.introspect import ActionFieldIntrospector
from llmr_updated_arch.integrations.krrood.match_reader import underspecified_match

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr_updated_arch.generation import Reasoner
    from llmr_updated_arch.hypotheses import BuildOrchestrator
    from llmr_updated_arch.integrations.pycram import PycramContext, PycramPlanNode


def instance_from_instruction(
    instruction: str,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[list["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> Any:
    """Classify *instruction*, build an underspecified match, and resolve it."""

    action_cls, classification = classify_instruction(
        instruction,
        llm=llm,
        action_registry=action_registry,
    )
    match = underspecified_match(action_cls, ActionFieldIntrospector())
    backend = _make_instruction_backend(
        instruction=instruction,
        llm=llm,
        symbol_type=symbol_type,
        classification=classification,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return next(iter(backend.evaluate(match)))


def plan_from_instruction(
    instruction: str,
    context: "PycramContext",
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[list["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> "PycramPlanNode":
    """Build a single executable PyCRAM plan node from a natural-language instruction."""

    from llmr_updated_arch.integrations.pycram import execute_single

    action_cls, classification = classify_instruction(
        instruction,
        llm=llm,
        action_registry=action_registry,
    )
    match = underspecified_match(action_cls, ActionFieldIntrospector())
    context.query_backend = _make_instruction_backend(
        instruction=instruction,
        llm=llm,
        symbol_type=symbol_type,
        classification=classification,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return execute_single(match, context)


def sequential_plan_from_instruction(
    instruction: str,
    context: "PycramContext",
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[list["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> list["PycramPlanNode"]:
    """Decompose a compound instruction and build one plan node per atomic step."""

    from llmr_updated_arch.generation.decomposer import TaskDecomposer

    decomposed = TaskDecomposer(llm=llm).decompose(instruction)
    return [
        plan_from_instruction(
            step,
            context,
            llm,
            symbol_type=symbol_type,
            action_registry=action_registry,
            reasoners=reasoners,
            sg_model_orchestrator=sg_model_orchestrator,
        )
        for step in decomposed.steps
    ]


def classify_instruction(
    instruction: str,
    llm: "BaseChatModel",
    action_registry: Optional[Dict[str, type]] = None,
) -> tuple[type, Any]:
    """Return the action class and classification result for *instruction*."""

    if action_registry is None:
        from llmr_updated_arch.integrations.pycram import discover_action_classes

        action_registry = discover_action_classes()

    classification = infer_action_class(
        instruction=instruction,
        llm=llm,
        action_registry=action_registry,
    )
    action_cls = (
        action_registry.get(classification.action_type) if classification else None
    )
    if action_cls is None:
        raise LLMActionClassificationFailed(instruction=instruction)
    return action_cls, classification


def _make_instruction_backend(
    instruction: str,
    llm: "BaseChatModel",
    symbol_type: type,
    classification: Any,
    reasoners: Optional[list["Reasoner"]],
    sg_model_orchestrator: Optional["BuildOrchestrator"],
) -> Any:
    from llmr_updated_arch.core.backend import LLMBackend

    return LLMBackend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        strict_required=True,
        classification=classification,
        reasoners=reasoners or [],
        sg_model_orchestrator=sg_model_orchestrator,
    )
