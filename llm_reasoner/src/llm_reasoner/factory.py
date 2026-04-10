"""
User-facing factory functions for NL-driven plan construction.

Two entry points:
  nl_plan()       — single atomic instruction → single PlanNode
  nl_sequential() — multi-step instruction → list of PlanNodes (decomposed)

Both functions wire up LLMBackend on the provided context, so the user
never has to instantiate the backend directly.

The caller must supply groundable_type (e.g. Body from SDT). This is the only
robot-framework type that crosses the boundary here — it is passed to LLMBackend
and never imported by this package.
"""
from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Dict, List, Optional

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pycram.datastructures.dataclasses import Context
    from pycram.plans.nodes import PlanNode
    from krrood.symbol_graph.symbol_graph import Symbol


def nl_plan(
    instruction: str,
    context: "Context",
    llm: "BaseChatModel",
    groundable_type: "type[Symbol]",
    action_registry: Optional[Dict[str, type]] = None,
    robot_context: Optional[Dict] = None,
) -> "PlanNode":
    """
    Build a single executable PlanNode from a natural-language instruction.

    Internally:
      1. LLM classifies the instruction → action class
      2. Builds a fully-underspecified Match (all fields set to ...)
      3. Sets LLMBackend as context.query_backend
      4. Returns execute_single(match, context)

    :param instruction:    The natural-language instruction.
    :param context:        PyCRAM Context (world + robot + query_backend).
    :param llm:            LangChain-compatible chat model.
    :param groundable_type: Symbol subclass for scene objects (e.g. Body from SDT).
                           Passed to LLMBackend for entity grounding; never imported here.
    :param action_registry: Optional {class_name: class} dict.
                           Auto-discovered from pycram if None.
    :param robot_context:  Optional dict injected into LLMBackend.context.
                           Put manipulators here: {"manipulators": {"LEFT": l, "RIGHT": r}}.
    :returns: A PlanNode ready to be performed.
    :raises ValueError: If the LLM cannot classify the action type.
    """
    from llm_reasoner.backend import LLMBackend
    from llm_reasoner.reasoning.slot_filler import classify_action
    from pycram.plans.factories import execute_single

    # Step 1: Classify action type from NL instruction
    action_cls = classify_action(
        instruction=instruction,
        llm=llm,
        action_registry=action_registry,
    )
    if action_cls is None:
        raise ValueError(
            f"Could not classify an action type from: {instruction!r}. "
            "Check that pycram action classes are importable and the LLM is reachable."
        )

    # Step 2: Build a fully-underspecified Match (all fields set to ...)
    match = _fully_underspecified(action_cls)

    # Step 3: Set LLMBackend on context
    context.query_backend = LLMBackend(
        instruction=instruction,
        llm=llm,
        groundable_type=groundable_type,
        context=robot_context or {},
    )

    # Step 4: Return the plan node
    return execute_single(match, context)


def nl_sequential(
    instruction: str,
    context: "Context",
    llm: "BaseChatModel",
    groundable_type: "type[Symbol]",
    action_registry: Optional[Dict[str, type]] = None,
    robot_context: Optional[Dict] = None,
) -> List["PlanNode"]:
    """
    Decompose a multi-step NL instruction and return one PlanNode per atomic step.

    Each step gets its own LLMBackend instance with a step-specific instruction,
    so LLM reasoning context is preserved per step.

    :param instruction:    The (possibly multi-step) natural-language instruction.
    :param context:        PyCRAM Context shared across all steps.
    :param llm:            LangChain-compatible chat model.
    :param groundable_type: Symbol subclass for scene objects (e.g. Body from SDT).
    :param action_registry: Optional action class registry (auto-discovered if None).
    :param robot_context:  Optional robot context dict forwarded to each LLMBackend.
    :returns: List of PlanNodes, one per atomic step, in execution order.
    """
    from llm_reasoner.reasoning.decomposer import TaskDecomposer

    decomposed = TaskDecomposer(llm=llm).decompose(instruction)
    return [
        nl_plan(
            step,
            context,
            llm,
            groundable_type=groundable_type,
            action_registry=action_registry,
            robot_context=robot_context,
        )
        for step in decomposed.steps
    ]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fully_underspecified(action_cls: type):
    """Build a Match for action_cls with ALL fields set to Ellipsis."""
    from krrood.entity_query_language.query.match import Match

    free_fields = _get_settable_fields(action_cls)
    match = Match(action_cls)
    if free_fields:
        match(**{name: ... for name in free_fields})
    return match


def _get_settable_fields(action_cls: type) -> List[str]:
    """Return names of all settable fields on an action class.

    Skips internal krrood/pycram bookkeeping fields that the LLM must not fill.
    """
    _SKIP_FIELDS = {"id", "plan_node"}
    if dataclasses.is_dataclass(action_cls):
        return [
            f.name
            for f in dataclasses.fields(action_cls)
            if not f.name.startswith("_") and f.name not in _SKIP_FIELDS
        ]
    try:
        sig = inspect.signature(action_cls.__init__)
        return [
            name
            for name, param in sig.parameters.items()
            if name != "self" and not name.startswith("_")
        ]
    except (TypeError, ValueError):
        return []
