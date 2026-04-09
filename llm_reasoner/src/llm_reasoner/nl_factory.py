"""
User-facing factory functions for NL-driven plan construction.

These are the primary entry points for users who want to drive the robot
purely from natural language — no knowledge of PyCRAM action classes,
Match expressions, or backend configuration required.

Two levels:
  nl_plan()       — single atomic instruction → single PlanNode
  nl_sequential() — multi-step instruction → list of PlanNodes (decomposed)

Both functions wire up LLMBackend on the provided context, so the user
never has to instantiate the backend directly.
"""
from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import List, Optional

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from pycram.datastructures.dataclasses import Context
    from pycram.plans.nodes import PlanNode


def nl_plan(
    instruction: str,
    context: "Context",
    llm: "BaseChatModel",
    action_registry: Optional[dict] = None,
) -> "PlanNode":
    """
    Build a single executable PlanNode from a natural-language instruction.

    Internally:
      1. LLM classifies the instruction → action class (e.g. PickUpAction)
      2. Builds a fully-underspecified Match (all fields set to ...)
      3. Sets LLMBackend as context.query_backend (carries the instruction)
      4. Returns execute_single(match, context)

    The user provides only natural language — no Match, no action class, no backend.

    Example::

        plan = nl_plan("pick up the milk from the table", context=context, llm=llm)
        plan.perform()

    :param instruction:     The natural-language instruction.
    :param context:         PyCRAM Context (world + robot + query_backend).
                            context.query_backend will be replaced with LLMBackend.
    :param llm:             A LangChain-compatible chat model.
    :param action_registry: Optional dict mapping class name → class.
                            Discovered automatically from pycram if not provided.
    :returns: A PlanNode ready to be performed.
    :raises ValueError: If the LLM cannot classify the action type.
    """
    from llm_reasoner.backend import LLMBackend
    from llm_reasoner.workflows.slot_filler import classify_action
    from pycram.plans.factories import execute_single

    # Step 1: Classify action type from NL instruction
    action_cls = classify_action(
        instruction=instruction,
        llm=llm,
        world=context.world,
        action_registry=action_registry,
    )
    if action_cls is None:
        raise ValueError(
            f"Could not classify an action type from instruction: {instruction!r}. "
            "Check that pycram action classes are importable and the LLM is reachable."
        )

    # Step 2: Build a fully-underspecified Match (all fields = ...)
    match = _fully_underspecified(action_cls)

    # Step 3: Set LLMBackend on context — the backend carries the instruction
    # and does the reasoning when UnderspecifiedNode calls backend.evaluate(match)
    context.query_backend = LLMBackend(
        instruction=instruction,
        llm=llm,
        world=context.world,
    )

    # Step 4: Return the plan node — mirrors existing PyCRAM usage pattern
    return execute_single(match, context)


def nl_sequential(
    instruction: str,
    context: "Context",
    llm: "BaseChatModel",
    action_registry: Optional[dict] = None,
) -> List["PlanNode"]:
    """
    Decompose a multi-step NL instruction and return one PlanNode per atomic step.

    Internally uses TaskDecomposer (LangGraph-based) to split the instruction
    into atomic steps, then calls nl_plan() for each step.

    Each step gets its own LLMBackend instance carrying the step-specific
    instruction — full LLM reasoning context is preserved per step.

    Example::

        plans = nl_sequential(
            "go to the table and pick up the milk, then put it in the fridge",
            context=context,
            llm=llm,
        )
        for plan in plans:
            plan.perform()

    :param instruction:     The (potentially multi-step) natural-language instruction.
    :param context:         PyCRAM Context shared across all steps.
    :param llm:             A LangChain-compatible chat model.
    :param action_registry: Optional action class registry (auto-discovered if None).
    :returns: List of PlanNodes, one per atomic step, in execution order.
    """
    from llm_reasoner.task_decomposer import TaskDecomposer

    decomposed = TaskDecomposer(llm=llm).decompose(instruction)
    return [
        nl_plan(step, context, llm, action_registry=action_registry)
        for step in decomposed.steps
    ]


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _fully_underspecified(action_cls: type):
    """
    Build a Match expression for the given action class with ALL fields set
    to Ellipsis, making them free for LLMBackend to fill.

    Mirrors the underspecified(ActionClass)(field=..., ...) pattern but
    infers all fields automatically from the class definition — the user
    never has to know which fields exist.
    """
    from krrood.entity_query_language.query.match import Match

    free_fields = _get_settable_fields(action_cls)

    match = Match(action_cls)
    if free_fields:
        match(**{name: ... for name in free_fields})
    return match


def _get_settable_fields(action_cls: type) -> List[str]:
    """
    Return the names of all settable fields on an action class.

    Handles both dataclasses (standard PyCRAM pattern) and plain classes.
    Skips internal KRROOD fields (prefixed with _ or named 'id').
    """
    # Prefer dataclass fields — most PyCRAM actions are dataclasses
    if dataclasses.is_dataclass(action_cls):
        return [
            f.name
            for f in dataclasses.fields(action_cls)
            if not f.name.startswith("_") and f.name != "id"
        ]

    # Fall back to __init__ signature for non-dataclass actions
    try:
        sig = inspect.signature(action_cls.__init__)
        return [
            name
            for name, param in sig.parameters.items()
            if name != "self" and not name.startswith("_")
        ]
    except (TypeError, ValueError):
        return []
