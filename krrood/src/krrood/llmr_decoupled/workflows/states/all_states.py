"""LangGraph state type definitions — generic, no robot/framework references."""

from __future__ import annotations

from typing_extensions import Any, Dict, List, Optional

from langgraph.graph import MessagesState


class SlotFillingState(MessagesState):
    """State for the Phase 1 slot-filling pipeline."""

    instruction: str
    """Raw natural-language instruction from the user."""

    world_context: str
    """Serialised world snapshot used as optional reference context for the LLM."""

    slot_schema: Optional[Dict[str, Any]]
    """Serialised ActionSlotSchema produced by the slot-filler LLM node."""

    error: Optional[str]
    """Non-fatal error or warning message from any node."""


class DiscreteResolutionState(MessagesState):
    """State for the Phase 2 discrete-resolution pipeline."""

    world_context: str
    known_parameters: str
    parameters_to_resolve: str

    resolved_schema: Optional[Dict[str, Any]]
    """Serialised resolution schema produced by the resolver LLM."""

    error: Optional[str]


class RecoveryState(MessagesState):
    """State for the recovery resolution pipeline."""

    world_context: str
    original_instruction: str
    failed_action_description: str
    error_message: str

    resolved_schema: Optional[Dict[str, Any]]
    """Serialised RecoverySchema produced by the recovery resolver LLM."""

    error: Optional[str]
