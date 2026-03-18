"""Phase 1 LangGraph agent: NL instruction → PickUpSlotSchema.

Graph structure:
    START → slot_filler_node → END

The single node calls the LLM with the slot-filling prompt and structured output
to produce a ``PickUpSlotSchema``.  The result is stored in the state as a plain
dict so it can be passed downstream without import cycles.

Usage::

    from generative_backend.workflows.agents.slot_filler import (
        slot_filler_graph,
        run_slot_filler,
    )

    schema = run_slot_filler("Pick up the red cup from the table with your left arm")
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph.state import END, START, StateGraph

from ..llm_configuration import default_llm
from ..prompts.pick_up_prompts import pick_up_slot_filler_prompt
from ..pydantics.pick_up_schemas import PickUpSlotSchema
from ..states.all_states import SlotFillingState

logger = logging.getLogger(__name__)

# ── Structured LLM client for this agent ──────────────────────────────────────

_slot_filler_llm = default_llm.with_structured_output(
    PickUpSlotSchema, method="function_calling"
)

# ── Graph nodes ────────────────────────────────────────────────────────────────


def slot_filler_node(state: SlotFillingState) -> Dict[str, Any]:
    """Call the LLM to extract slot values from the instruction.

    Produces a ``PickUpSlotSchema`` and stores it as a plain dict in the state.
    On LLM failure the node logs the error and stores it in ``state['error']``
    so the pipeline can decide how to handle it.
    """
    instruction: str = state["instruction"]
    world_context: str = state.get("world_context", "")

    chain = pick_up_slot_filler_prompt | _slot_filler_llm

    try:
        schema: PickUpSlotSchema = chain.invoke(
            {"instruction": instruction, "world_context": world_context}
        )
        logger.debug(
            "Slot filler output – object='%s', arm=%s, grasp=%s",
            schema.object_description.name,
            schema.arm,
            schema.grasp_params,
        )
        return {"slot_schema": schema.model_dump(), "error": None}

    except Exception as exc:  # noqa: BLE001
        logger.error("Slot filler LLM call failed: %s", exc)
        return {"slot_schema": None, "error": str(exc)}


# ── Graph assembly ─────────────────────────────────────────────────────────────

_builder = StateGraph(SlotFillingState)
_builder.add_node("slot_filler", slot_filler_node)
_builder.add_edge(START, "slot_filler")
_builder.add_edge("slot_filler", END)

slot_filler_graph = _builder.compile()


# ── Convenience helper ─────────────────────────────────────────────────────────


def run_slot_filler(
    instruction: str,
    world_context: str = "",
) -> PickUpSlotSchema | None:
    """Run the slot-filler graph synchronously and return the schema.

    Returns ``None`` if the LLM call failed (error is logged).

    :param instruction: Raw NL instruction.
    :param world_context: Optional serialised world snapshot for context.
    :return: Parsed ``PickUpSlotSchema`` or ``None`` on failure.
    """
    final_state = slot_filler_graph.invoke(
        {"instruction": instruction, "world_context": world_context}
    )

    if final_state.get("error"):
        logger.warning("run_slot_filler: %s", final_state["error"])
        return None

    raw = final_state.get("slot_schema")
    if raw is None:
        return None

    return PickUpSlotSchema.model_validate(raw)
