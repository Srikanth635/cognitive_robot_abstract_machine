"""User-facing factory functions for NL-driven plan construction via sg_model.

All pycram access goes through llmr.pycram.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Dict, List, Optional

from krrood.symbol_graph.symbol_graph import Symbol
from llmr.bridge.introspect import ActionFieldIntrospector
from llmr.bridge.match_reader import underspecified_match
from llmr.exceptions import LLMActionClassificationFailed
from llmr.pycram import PycramContext, PycramPlanNode, execute_single

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr.reasoning import Reasoner
    from llmr.schemas import ActionClassificationResult as ActionClassification
    from llmr.hypotheses import BuildOrchestrator


def plan_from_instruction(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[List["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> PycramPlanNode:
    """
    Build a single executable PlanNode from a natural-language instruction.

    Internally:
      1. LLM classifies the instruction -> action class
      2. Builds an underspecified Match for required schema fields
      3. Sets a strict sg_model-backed LLMBackend as `context.query_backend`
      4. Returns execute_single(match, context)

    :param instruction: The natural-language instruction.
    :param context: PyCRAM Context carrying `query_backend`.
    :param llm: LangChain-compatible chat model.
    :param symbol_type: Symbol subclass for scene objects.
                        Passed to LLMBackend for entity grounding.
    :param action_registry: Optional {class_name: class} dict.
                            Auto-discovered by the PyCRAM bridge if None.
    :param reasoners: Optional pluggable llmr reasoners to run after slot filling.
    :param sg_model_orchestrator: Optional shared sg_model orchestrator used to
                                  accumulate reasoner sidecars into one
                                  object-domain hypothesis graph.
    :returns: A PlanNode ready to be performed.
    :raises ValueError: If the LLM cannot classify the action type.
    """
    from llmr.pycram import discover_action_classes
    from llmr.reasoning.slot_filler import infer_action_class

    if action_registry is None:
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

    match = underspecified_match(action_cls, ActionFieldIntrospector())

    context.query_backend = _make_llm_backend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        strict_required=True,
        classification=classification,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return execute_single(match, context)


def sequential_plan_from_instruction(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[List["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> List[PycramPlanNode]:
    """
    Decompose a multi-step NL instruction and return one PlanNode per atomic step.

    Each step gets its own backend instance with a step-specific instruction, so
    LLM reasoning context is preserved per step.
    """
    from llmr.reasoning.decomposer import TaskDecomposer

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


def plan_from_match(
    match: Any,
    context: PycramContext,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
    reasoners: Optional[List["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> PycramPlanNode:
    """Return a PlanNode for an already-built underspecified Match."""
    context.query_backend = _make_llm_backend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return execute_single(match, context)


def instance_from_match(
    match: Any,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
    reasoners: Optional[List["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> Any:
    """Resolve an underspecified Match and return the concrete action instance."""
    backend = _make_llm_backend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners,
        sg_model_orchestrator=sg_model_orchestrator,
    )
    return next(iter(backend.evaluate(match)))


def _make_llm_backend(
    llm: "BaseChatModel",
    symbol_type: type,
    instruction: Optional[str],
    strict_required: bool,
    world_context_provider: Optional[Callable[[], str]] = None,
    classification: Optional["ActionClassification"] = None,
    reasoners: Optional[List["Reasoner"]] = None,
    sg_model_orchestrator: Optional["BuildOrchestrator"] = None,
) -> Any:
    """Create the sg_model-backed backend used by factory entry points."""
    from llmr.backend import LLMBackend

    return LLMBackend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        classification=classification,
        reasoners=reasoners or [],
        sg_model_orchestrator=sg_model_orchestrator,
    )
