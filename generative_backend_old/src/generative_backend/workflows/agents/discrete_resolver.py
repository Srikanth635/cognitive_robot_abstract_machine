"""Phase 2 LangGraph agent: world context + partial params → PickUpDiscreteResolutionSchema.

Graph structure:
    START → discrete_resolver_node → END

The node receives a world context string (built by ``WorldContextBuilder``),
a summary of already-known parameters, and a description of what still needs to
be resolved.  It calls the LLM and returns a ``PickUpDiscreteResolutionSchema``
with all discrete fields filled.

Usage::

    from generative_backend.workflows.agents.discrete_resolver import (
        run_discrete_resolver,
    )

    resolution = run_discrete_resolver(
        world_context="...",
        known_parameters="arm=LEFT",
        parameters_to_resolve="approach_direction, vertical_alignment, rotate_gripper",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph.state import END, START, StateGraph

from ..llm_configuration import default_llm
from ..prompts.pick_up import pick_up_resolver_prompt
from ..schemas.pick_up import PickUpDiscreteResolutionSchema
from ..states.all_states import DiscreteResolutionState

logger = logging.getLogger(__name__)

# ── Structured LLM client for this agent ──────────────────────────────────────

_resolver_llm = default_llm.with_structured_output(
    PickUpDiscreteResolutionSchema, method="function_calling"
)

# ── Graph nodes ────────────────────────────────────────────────────────────────


def discrete_resolver_node(state: DiscreteResolutionState) -> Dict[str, Any]:
    """Call the LLM to reason about and fill discrete grasp parameters.

    Stores the result as a plain dict in ``state['resolved_schema']``.
    """
    world_context: str = state["world_context"]
    known_parameters: str = state["known_parameters"]
    parameters_to_resolve: str = state["parameters_to_resolve"]

    chain = pick_up_resolver_prompt | _resolver_llm

    try:
        schema: PickUpDiscreteResolutionSchema = chain.invoke(
            {
                "world_context": world_context,
                "known_parameters": known_parameters,
                "parameters_to_resolve": parameters_to_resolve,
            }
        )
        logger.debug(
            "Discrete resolver output – arm=%s, approach=%s, vertical=%s, "
            "rotate=%s | reasoning: %s",
            schema.arm,
            schema.approach_direction,
            schema.vertical_alignment,
            schema.rotate_gripper,
            schema.reasoning,
        )
        return {"resolved_schema": schema.model_dump(), "error": None}

    except Exception as exc:  # noqa: BLE001
        logger.error("Discrete resolver LLM call failed: %s", exc)
        return {"resolved_schema": None, "error": str(exc)}


# ── Graph assembly ─────────────────────────────────────────────────────────────

_builder = StateGraph(DiscreteResolutionState)
_builder.add_node("discrete_resolver", discrete_resolver_node)
_builder.add_edge(START, "discrete_resolver")
_builder.add_edge("discrete_resolver", END)

discrete_resolver_graph = _builder.compile()


# ── Convenience helper ─────────────────────────────────────────────────────────


def run_discrete_resolver(
    world_context: str,
    known_parameters: str,
    parameters_to_resolve: str,
) -> PickUpDiscreteResolutionSchema | None:
    """Run the discrete resolver graph synchronously.

    :param world_context: Serialised world snapshot for scene reasoning.
    :param known_parameters: Human-readable summary of known slot values.
    :param parameters_to_resolve: Names of the parameters still to be decided.
    :return: ``PickUpDiscreteResolutionSchema`` or ``None`` on failure.
    """
    final_state = discrete_resolver_graph.invoke(
        {
            "world_context": world_context,
            "known_parameters": known_parameters,
            "parameters_to_resolve": parameters_to_resolve,
        }
    )

    if final_state.get("error"):
        logger.warning("run_discrete_resolver: %s", final_state["error"])
        return None

    raw = final_state.get("resolved_schema")
    if raw is None:
        return None

    return PickUpDiscreteResolutionSchema.model_validate(raw)
