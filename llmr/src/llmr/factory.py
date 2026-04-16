"""User-facing factory functions for NL-driven plan construction.

All pycram access goes through llmr.pycram_bridge.
"""
from __future__ import annotations

import typing
from typing_extensions import Any, Callable, Dict, List, Optional

from krrood.symbol_graph.symbol_graph import Symbol
from llmr.exceptions import LLMActionClassificationFailed
from llmr.pycram_bridge import PycramContext, PycramPlanNode, execute_single

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from krrood.entity_query_language.query.match import Match


def nl_plan(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
) -> PycramPlanNode:
    """
    Build a single executable PlanNode from a natural-language instruction.

    Internally:
      1. LLM classifies the instruction → action class
      2. Builds an underspecified Match for required schema fields
      3. Sets a strict LLMBackend as context.query_backend
      4. Returns execute_single(match, context)

    :param instruction:    The natural-language instruction.
    :param context:        PyCRAM Context (world + robot + query_backend).
    :param llm:            LangChain-compatible chat model.
    :param groundable_type: Symbol subclass for scene objects.
                           Passed to LLMBackend for entity grounding; never imported here.
    :param action_registry: Optional {class_name: class} dict.
                           Auto-discovered by the PyCRAM bridge if None.
    :returns: A PlanNode ready to be performed.
    :raises ValueError: If the LLM cannot classify the action type.
    """
    from llmr.backend import LLMBackend
    from llmr.reasoning.slot_filler import classify_action

    # Step 1: Classify action type from NL instruction
    action_cls = classify_action(
        instruction=instruction,
        llm=llm,
        action_registry=action_registry,
    )
    if action_cls is None:
        raise LLMActionClassificationFailed(instruction=instruction)

    # Step 2: Build an underspecified Match for required schema fields.
    match = _fully_underspecified(action_cls)

    # Step 3: Set strict LLMBackend on context.
    context.query_backend = LLMBackend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        strict_required=True,
    )

    # Step 4: Return the plan node
    return execute_single(match, context)


def nl_sequential(
    instruction: str,
    context: PycramContext,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    action_registry: Optional[Dict[str, type]] = None,
) -> List[PycramPlanNode]:
    """
    Decompose a multi-step NL instruction and return one PlanNode per atomic step.

    Each step gets its own LLMBackend instance with a step-specific instruction,
    so LLM reasoning context is preserved per step.

    :param instruction:    The (possibly multi-step) natural-language instruction.
    :param context:        PyCRAM Context shared across all steps.
    :param llm:            LangChain-compatible chat model.
    :param groundable_type: Symbol subclass for scene objects.
    :param action_registry: Optional action class registry (auto-discovered if None).
    :returns: List of PlanNodes, one per atomic step, in execution order.
    """
    from llmr.reasoning.decomposer import TaskDecomposer

    decomposed = TaskDecomposer(llm=llm).decompose(instruction)
    return [
        nl_plan(
            step,
            context,
            llm,
            groundable_type=groundable_type,
            action_registry=action_registry,
        )
        for step in decomposed.steps
    ]


def resolve_match(
    match: Any,
    context: PycramContext,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
) -> PycramPlanNode:
    """Return a PlanNode for an already-built underspecified Match.

    Plan helper for Role 2.  For a non-executing resolved action instance, use
    ``resolve_params``.

    :param match:           A fully or partially underspecified krrood Match expression.
    :param context:         PyCRAM Context (world + robot).
    :param llm:             LangChain-compatible chat model.
    :param groundable_type: Symbol subclass scoping entity grounding. Defaults to Symbol.
    :param instruction:     Optional NL hint included in the slot-filler prompt.
                            Omit when the action type and fixed slots carry the intent.
    :param world_context_provider: Optional callable returning runtime world context text.
    :param strict_required: Raise if required fields remain unresolved before construction.
    :returns: A PlanNode ready to be performed.
    """
    from llmr.backend import LLMBackend
    context.query_backend = LLMBackend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
    )
    return execute_single(match, context)


def resolve_params(
    match: Any,
    llm: "BaseChatModel",
    groundable_type: type = Symbol,
    instruction: Optional[str] = None,
    world_context_provider: Optional[Callable[[], str]] = None,
    strict_required: bool = False,
) -> Any:
    """Resolve an underspecified Match and return the concrete action instance.

    Role 2 non-executing API: no action classification, no Match construction, and no
    PyCRAM PlanNode creation.  The supplied Match is still updated by the backend as
    part of normal KRROOD evaluation.
    """
    from llmr.backend import LLMBackend

    backend = LLMBackend(
        llm=llm,
        groundable_type=groundable_type,
        instruction=instruction,
        world_context_provider=world_context_provider,
        strict_required=strict_required,
    )
    return next(iter(backend.evaluate(match)))


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fully_underspecified(action_cls: type) -> "Match[Any]":
    """Return a Match(*action_cls*) with required schema fields set to ``...``."""
    from krrood.entity_query_language.query.match import Match

    free_kwargs = _get_required_schema_kwargs(action_cls)
    if free_kwargs is None:
        free_fields = _get_settable_fields(action_cls)
        free_kwargs = {name: ... for name in free_fields}

    match = Match(action_cls)
    if free_kwargs:
        match(**free_kwargs)
    return match


def _get_required_schema_kwargs(action_cls: type) -> Optional[Dict[str, Any]]:
    """Return required Match kwargs, nesting complex fields when possible."""
    try:
        from llmr.pycram_bridge import PycramIntrospector

        schema = PycramIntrospector().introspect(action_cls)
    except Exception:
        return None

    return _required_kwargs_from_field_specs(schema.fields)


def _required_kwargs_from_field_specs(fields: List[Any]) -> Dict[str, Any]:
    """Build ``Match`` kwargs for required fields from introspected FieldSpecs."""
    return {
        field.name: _underspecified_value_for_field(field)
        for field in fields
        if _is_schema_field_settable(field, required_only=True)
    }


def _underspecified_value_for_field(field: Any) -> Any:
    """Return ``...`` or a nested Match for a required FieldSpec."""
    from krrood.entity_query_language.query.match import Match
    from llmr.pycram_bridge.introspector import FieldKind

    if field.kind != FieldKind.COMPLEX:
        return ...

    nested_kwargs = _required_kwargs_from_field_specs(field.sub_fields)
    nested_match = Match(field.raw_type)
    nested_match(**nested_kwargs)
    return nested_match


def _get_settable_fields(action_cls: type) -> List[str]:
    """Return names of all settable fields on an action class."""
    return _get_schema_fields(action_cls, required_only=False) or []


def _get_required_schema_fields(action_cls: type) -> Optional[List[str]]:
    """Return required action fields from PycramIntrospector, or None on failure."""
    return _get_schema_fields(action_cls, required_only=True)


def _get_schema_fields(action_cls: type, required_only: bool) -> Optional[List[str]]:
    """Return prompt-settable field names from the KRROOD-backed action schema."""
    try:
        from llmr.pycram_bridge import PycramIntrospector

        schema = PycramIntrospector().introspect(action_cls)
    except Exception:
        return None

    return [
        field.name
        for field in schema.fields
        if _is_schema_field_settable(field, required_only=required_only)
    ]


def _is_schema_field_settable(field: Any, required_only: bool) -> bool:
    """Return whether a schema field should be exposed as LLM-settable."""
    return (
        (not required_only or not field.is_optional)
        and not field.name.startswith("_")
    )
