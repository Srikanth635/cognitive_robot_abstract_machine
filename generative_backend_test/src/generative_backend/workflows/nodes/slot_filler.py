"""Phase 1 LangGraph node: NL instruction → typed per-action slot schema.

Graph structure:
    START → slot_filler_node → END

The node makes a single LLM call that:
  1. Classifies the instruction into a supported action type.
  2. Extracts all relevant slot parameters for that action type.

Internally the LLM fills a private intermediate schema (``_SlotFillerOutput``)
that carries the ``action_type`` discriminator plus every possible field across
all supported actions.  After the LLM call the node projects this into the
correct, tightly typed, action-specific schema:

    PickUpAction instruction  →  ``PickUpSlotSchema``
    PlaceAction  instruction  →  ``PlaceSlotSchema``

This keeps the public interface clean — callers always receive a schema that
has *only* the fields relevant to the classified action, with no optional fields
from other action types leaking through.

Usage::

    from generative_backend.workflows.nodes.slot_filler import run_slot_filler

    schema = run_slot_filler("Pick up the red cup from the table")
    # → PickUpSlotSchema

    schema = run_slot_filler("Place the mug on the kitchen counter")
    # → PlaceSlotSchema
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Union

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..llm_configuration import default_llm
from ..prompts.slot_filler import slot_filler_prompt
from ..schemas.common import EntityDescriptionSchema
from ..schemas.pick_up import GraspParamsSchema, PickUpSlotSchema
from ..schemas.place import PlaceSlotSchema
from ..states.all_states import SlotFillingState

logger = logging.getLogger(__name__)


# ── Public type alias ──────────────────────────────────────────────────────────

ActionSlotSchema = Union[PickUpSlotSchema, PlaceSlotSchema]
"""The return type of ``run_slot_filler``.

Callers can use ``isinstance(schema, PickUpSlotSchema)`` or check
``schema.action_type`` to branch on the action type.
"""


# ── Private intermediate schema ────────────────────────────────────────────────


class _SlotFillerOutput(BaseModel):
    """Private schema used only inside this node for the LLM structured output call.

    The LLM fills this flat schema in one call; we then project it onto the
    correct typed per-action schema.  This class is never exposed outside the
    node — all public functions return ``PickUpSlotSchema`` or ``PlaceSlotSchema``.
    """

    action_type: Literal["PickUpAction", "PlaceAction"] = Field(
        description="The action type this instruction maps to."
    )
    object_description: EntityDescriptionSchema = Field(
        description="The primary object: to pick up (PickUpAction) or to place (PlaceAction)."
    )
    arm: Optional[Literal["LEFT", "RIGHT", "BOTH"]] = Field(
        default=None,
        description="Which arm to use.  Null unless the instruction explicitly names one.",
    )
    # PickUpAction only
    grasp_params: Optional[GraspParamsSchema] = Field(
        default=None,
        description="Grasp configuration.  Only for PickUpAction; null for PlaceAction.",
    )
    # PlaceAction only
    target_description: Optional[EntityDescriptionSchema] = Field(
        default=None,
        description="Target placement location.  Only for PlaceAction; null for PickUpAction.",
    )


def _to_typed_schema(raw: _SlotFillerOutput) -> ActionSlotSchema:
    """Project the intermediate LLM output onto the correct per-action schema."""
    if raw.action_type == "PickUpAction":
        return PickUpSlotSchema(
            object_description=raw.object_description,
            arm=raw.arm,
            grasp_params=raw.grasp_params,
        )
    elif raw.action_type == "PlaceAction":
        return PlaceSlotSchema(
            object_description=raw.object_description,
            target_description=raw.target_description,
            arm=raw.arm,
        )
    else:  # future-proof guard
        raise ValueError(f"Unrecognised action_type from LLM: {raw.action_type!r}")


# ── LLM binding ───────────────────────────────────────────────────────────────

_slot_filler_llm = default_llm.with_structured_output(
    _SlotFillerOutput, method="function_calling"
)


# ── LangGraph node ────────────────────────────────────────────────────────────


def slot_filler_node(state: SlotFillingState) -> Dict[str, Any]:
    """LangGraph node: fills ``_SlotFillerOutput`` then projects to typed schema."""
    instruction: str = state["instruction"]
    world_context: str = state.get("world_context", "")

    chain = slot_filler_prompt | _slot_filler_llm
    try:
        raw: _SlotFillerOutput = chain.invoke(
            {"instruction": instruction, "world_context": world_context}
        )
        typed = _to_typed_schema(raw)
        logger.debug(
            "slot_filler_node – action_type=%s, object=%s",
            typed.action_type,
            typed.object_description.name,
        )
        return {"slot_schema": typed.model_dump(), "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.error("slot_filler_node error: %s", exc, exc_info=True)
        return {"slot_schema": None, "error": str(exc)}


# ── LangGraph graph ───────────────────────────────────────────────────────────

_builder: StateGraph = StateGraph(SlotFillingState)
_builder.add_node("slot_filler", slot_filler_node)
_builder.add_edge(START, "slot_filler")
_builder.add_edge("slot_filler", END)

slot_filler_graph = _builder.compile()


# ── Public entry point ────────────────────────────────────────────────────────


def run_slot_filler(
    instruction: str,
    world_context: str = "",
) -> Optional[ActionSlotSchema]:
    """Run the slot-filler node and return a typed per-action schema.

    :param instruction: Natural language robot instruction.
    :param world_context: Optional serialised world state string for context.
    :return: ``PickUpSlotSchema`` or ``PlaceSlotSchema`` on success; ``None`` on failure.
    """
    final_state = slot_filler_graph.invoke(
        {"instruction": instruction, "world_context": world_context}
    )

    if final_state.get("error") or final_state.get("slot_schema") is None:
        logger.warning("run_slot_filler: %s", final_state.get("error"))
        return None

    raw_dict: dict = final_state["slot_schema"]
    action_type = raw_dict.get("action_type")

    if action_type == "PickUpAction":
        return PickUpSlotSchema.model_validate(raw_dict)
    elif action_type == "PlaceAction":
        return PlaceSlotSchema.model_validate(raw_dict)
    else:
        logger.error("run_slot_filler: unexpected action_type %r in state.", action_type)
        return None
