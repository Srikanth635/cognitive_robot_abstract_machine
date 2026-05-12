"""User-facing factory functions for NL-driven plan construction.

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
    from llmr.hypotheses.projection import ProjectionOrchestrator as HypothesisGraphManager
    from llmr.reasoning import Reasoner
    from llmr.schemas import ActionClassificationResult as ActionClassification


def plan_from_instruction(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[List["Reasoner"]] = None,
    hypothesis_graph_manager: Optional["HypothesisGraphManager"] = None,
) -> PycramPlanNode:
    """
    Build a single executable PlanNode from a natural-language instruction.

    Internally:
      1. LLM classifies the instruction → action class
      2. Builds an underspecified Match for required schema fields
      3. Sets a strict LLMBackend as `context.query_backend`
      4. Returns execute_single(match, context)

    :param instruction:    The natural-language instruction.
    :param context:        PyCRAM Context carrying `query_backend`.
    :param llm:            LangChain-compatible chat model.
    :param symbol_type: Symbol subclass for scene objects.
                           Passed to LLMBackend for entity grounding; never imported here.
    :param action_registry: Optional {class_name: class} dict.
                           Auto-discovered by the PyCRAM bridge if None.
    :param reasoners: Optional pluggable llmr reasoners to run after slot filling.
    :param hypothesis_graph_manager: Optional orchestrator that projects reasoner sidecars
                           into the epistemic hypothesis graph after action construction.
    :returns: A PlanNode ready to be performed.
    :raises ValueError: If the LLM cannot classify the action type.
    """
    from llmr.pycram import discover_action_classes
    from llmr.reasoning.slot_filler import infer_action_class

    # Step 1: Classify action type from NL instruction.  infer_action_class returns
    # the raw ActionClassification; we resolve it to a concrete class here using
    # the same registry the classifier saw.
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

    # Step 2: Build an underspecified Match for required schema fields.
    match = underspecified_match(action_cls, ActionFieldIntrospector())

    # Step 3: Set strict LLMBackend on context.  The classification rides along
    # as a kwarg so the backend itself populates ``semantics.classification``
    # during ``_evaluate`` — no external mutation of ``backend.semantics``.
    context.query_backend = _make_llm_backend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        strict_required=True,
        classification=classification,
        reasoners=reasoners,
        hypothesis_graph_manager=hypothesis_graph_manager,
    )

    # Step 4: Return the plan node
    return execute_single(match, context)


def sequential_plan_from_instruction(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    symbol_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
    reasoners: Optional[List["Reasoner"]] = None,
    hypothesis_graph_manager: Optional["HypothesisGraphManager"] = None,
) -> List[PycramPlanNode]:
    """
    Decompose a multi-step NL instruction and return one PlanNode per atomic step.

    Each step gets its own LLMBackend instance with a step-specific instruction,
    so LLM reasoning context is preserved per step.

    :param instruction:    The (possibly multi-step) natural-language instruction.
    :param context:        PyCRAM Context shared across all steps.
    :param llm:            LangChain-compatible chat model.
    :param symbol_type: Symbol subclass for scene objects.
    :param action_registry: Optional action class registry (auto-discovered if None).
    :param reasoners: Optional pluggable llmr reasoners applied to every step backend.
    :param hypothesis_graph_manager: Optional manager shared across per-step backends.
    :returns: List of PlanNodes, one per atomic step, in execution order.
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
            hypothesis_graph_manager=hypothesis_graph_manager,
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
    hypothesis_graph_manager: Optional["HypothesisGraphManager"] = None,
) -> PycramPlanNode:
    """Return a PlanNode for an already-built underspecified Match.

    Plan helper for Role 2.  For a non-executing resolved action instance, use
    `instance_from_match()`.

    :param match:           A fully or partially underspecified krrood Match expression.
    :param context:         PyCRAM Context (world + robot).
    :param llm:             LangChain-compatible chat model.
    :param symbol_type: Symbol subclass scoping entity grounding. Defaults to Symbol.
    :param instruction:     Optional NL hint included in the slot-filler prompt.
                            Omit when the action type and fixed slots carry the intent.
    :param world_context_provider: Optional callable returning runtime world context text.
    :param strict_required: Raise if required fields remain unresolved before construction.
    :param reasoners: Optional pluggable llmr reasoners to run after slot filling.
    :param hypothesis_graph_manager: Optional orchestrator that projects reasoner sidecars
                                     into the epistemic hypothesis graph.
    :returns: A PlanNode ready to be performed.
    """
    context.query_backend = _make_llm_backend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners,
        hypothesis_graph_manager=hypothesis_graph_manager,
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
    hypothesis_graph_manager: Optional["HypothesisGraphManager"] = None,
) -> Any:
    """Resolve an underspecified Match and return the concrete action instance.

    Role 2 non-executing API: no action classification, no Match construction, and no
    PyCRAM PlanNode creation.  The supplied Match is still updated by the backend as
    part of normal KRROOD evaluation.
    """
    backend = _make_llm_backend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        reasoners=reasoners,
        hypothesis_graph_manager=hypothesis_graph_manager,
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
    hypothesis_graph_manager: Optional["HypothesisGraphManager"] = None,
) -> Any:
    """Create the LLM backend used by factory entry points."""
    from llmr.backend import LLMBackend

    return LLMBackend(
        llm=llm,
        symbol_type=symbol_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
        classification=classification,
        reasoners=reasoners or [],
        hypothesis_graph_manager=hypothesis_graph_manager,
    )
