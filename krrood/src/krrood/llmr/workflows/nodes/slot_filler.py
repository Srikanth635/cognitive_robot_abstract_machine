"""Slot-filler LangGraph node — generic, action types injected at runtime.

Supports two modes:
  - Plain dict mode: ``run_slot_filler(action_types={...})``
  - Schema mode:     ``run_slot_filler(action_schemas=[...])`` using pycram introspection
"""

from __future__ import annotations

import logging
from typing_extensions import Any, Dict, List, Optional, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from krrood.llmr.workflows.llm_configuration import default_llm
from krrood.llmr.workflows.prompts.slot_filler import (
    build_slot_filler_prompt,
    build_slot_filler_prompt_from_schemas,
)
from krrood.llmr.workflows.schemas.common import ActionSlotSchema, EntityDescriptionSchema
from krrood.llmr.workflows.states.all_states import SlotFillingState

if TYPE_CHECKING:
    from krrood.llmr.pycram_bridge.introspector import ActionSchema

logger = logging.getLogger(__name__)


# ── LLM output schema (matches new ActionSlotSchema) ──────────────────────────


class _SlotFillerOutput(BaseModel):
    """What the LLM returns — maps 1-to-1 onto ActionSlotSchema."""

    action_type: str = Field(
        description="The action type string exactly as listed in the supported actions."
    )
    entities: List[EntityDescriptionSchema] = Field(
        default_factory=list,
        description=(
            "World entities to ground. Each must have a 'role' matching the "
            "exact pycram constructor parameter name."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Non-entity parameters keyed by their pycram field name.",
    )
    manner: Optional[str] = Field(
        default=None,
        description="Execution style hint ('carefully', 'gently'). Null if not mentioned.",
    )
    constraints: Optional[List[str]] = Field(
        default=None,
        description="Explicit constraints from the instruction. Null if none.",
    )


def _to_action_slot_schema(raw: _SlotFillerOutput) -> ActionSlotSchema:
    return ActionSlotSchema(
        action_type=raw.action_type,
        entities=raw.entities,
        parameters=raw.parameters,
        manner=raw.manner,
        constraints=raw.constraints,
    )


# ── LLM binding (lazy singleton) ──────────────────────────────────────────────

_slot_filler_llm: Optional[Any] = None


def _get_slot_filler_llm() -> Any:
    global _slot_filler_llm
    if _slot_filler_llm is None:
        _slot_filler_llm = default_llm.with_structured_output(
            _SlotFillerOutput, method="function_calling"
        )
    return _slot_filler_llm


# ── LangGraph graph (cached per prompt config) ─────────────────────────────────

_graph_cache: Dict[str, Any] = {}


def clear_slot_filler_cache() -> None:
    """Clear the compiled LangGraph cache (call after prompt changes)."""
    _graph_cache.clear()


def _build_slot_filler_graph(prompt: Any) -> Any:
    """Build (or return cached) LangGraph for the given prompt template."""
    cache_key = str(prompt)
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    def slot_filler_node(state: SlotFillingState) -> Dict[str, Any]:
        instruction: str = state["instruction"]
        world_context: str = state.get("world_context", "")
        chain = prompt | _get_slot_filler_llm()
        try:
            raw: _SlotFillerOutput = chain.invoke(
                {"instruction": instruction, "world_context": world_context}
            )
            typed = _to_action_slot_schema(raw)
            logger.debug(
                "slot_filler_node – action_type=%s entities=%s",
                typed.action_type,
                [e.role for e in typed.entities],
            )
            return {"slot_schema": typed.model_dump(), "error": None}
        except Exception as exc:
            logger.error("slot_filler_node error: %s", exc, exc_info=True)
            return {"slot_schema": None, "error": str(exc)}

    builder = StateGraph(SlotFillingState)
    builder.add_node("slot_filler", slot_filler_node)
    builder.add_edge(START, "slot_filler")
    builder.add_edge("slot_filler", END)
    compiled = builder.compile()
    _graph_cache[cache_key] = compiled
    return compiled


# ── Public entry point ─────────────────────────────────────────────────────────


def run_slot_filler(
    instruction: str,
    world_context: str = "",
    action_types: Optional[Dict[str, str]] = None,
    action_schemas: Optional["List[ActionSchema]"] = None,
) -> Optional[ActionSlotSchema]:
    """Run the slot-filler and return an :class:`ActionSlotSchema`.

    :param instruction: Natural language instruction.
    :param world_context: Optional serialised world state string.
    :param action_types: ``{action_type: description}`` dict (plain mode).
    :param action_schemas: List of :class:`ActionSchema` objects from pycram
        introspection (richer mode — takes priority over *action_types*).
    :return: :class:`ActionSlotSchema` on success; ``None`` on failure.
    """
    if action_schemas:
        prompt = build_slot_filler_prompt_from_schemas(action_schemas)
    else:
        prompt = build_slot_filler_prompt(action_types or {})

    graph = _build_slot_filler_graph(prompt)
    final_state = graph.invoke(
        {"instruction": instruction, "world_context": world_context}
    )

    if final_state.get("error") or final_state.get("slot_schema") is None:
        logger.warning("run_slot_filler failed: %s", final_state.get("error"))
        return None

    try:
        schema = ActionSlotSchema.model_validate(final_state["slot_schema"])
        schema = _strip_implicit_parameters(schema, instruction)
        return schema
    except Exception as exc:
        logger.error("run_slot_filler: schema validation failed: %s", exc)
        return None


def _strip_implicit_parameters(
    schema: ActionSlotSchema, instruction: str
) -> ActionSlotSchema:
    """Remove parameters whose value does not appear verbatim in the instruction.

    The slot-filler must only extract what is explicitly stated.  This filter
    removes any parameter the LLM invented as a "reasonable default" — those
    will be resolved later by AutoActionHandler / EnumResolver.

    A parameter is kept only if its string value (case-insensitive) appears
    somewhere in the instruction text.
    """
    if not schema.parameters:
        return schema

    instruction_lower = instruction.lower()
    kept = {
        k: v
        for k, v in schema.parameters.items()
        if str(v).lower() in instruction_lower
    }
    if len(kept) != len(schema.parameters):
        dropped = set(schema.parameters) - set(kept)
        logger.debug(
            "_strip_implicit_parameters: dropped %s (not in instruction).", dropped
        )
        return schema.model_copy(update={"parameters": kept})
    return schema
