"""
Pydantic schemas for the LLM reasoning output.

Design difference from llmr/workflows/schemas/:
  - llmr had per-action-type schemas (PickUpSlotSchema, PlaceSlotSchema)
    with hardcoded field names — adding a new action required adding a new schema.
  - Here we use a single generic ActionReasoningOutput with a list of SlotValues.
    The LLM resolves whatever free slots the Match expression declares,
    regardless of action type. No schema changes needed for new action types.
"""
from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel

from llm_reasoner.workflows.schemas.common import EntityDescriptionSchema


class SlotValue(BaseModel):
    """
    A single resolved slot value produced by the LLM reasoning step.

    For entity slots (object_designator, target, etc.) the LLM provides
    both a raw value (the world body name) and an EntityDescriptionSchema
    capturing its reasoning about which entity was meant.

    For parameter slots (arm, grasp_type, approach_direction, etc.) only
    the value field is populated.
    """

    field_name: str
    """Name of the Match field being resolved. Must match an attribute name on the action class."""

    value: Any
    """
    The resolved concrete value.
    - For entity slots: the world body name as a string (e.g. "milk_1").
      LLMBackend will look this up in the world to get the actual Body object.
    - For enum/parameter slots: the enum member name or primitive value
      (e.g. "LEFT", "FRONT", True).
    """

    entity_description: Optional[EntityDescriptionSchema] = None
    """
    For entity slots: the LLM's semantic description of the entity it resolved.
    Kept for traceability and debugging — not used in action construction.
    """

    reasoning: str = ""
    """Per-slot explanation of why this value was chosen."""


class ActionReasoningOutput(BaseModel):
    """
    Structured output from the LLM reasoning step inside LLMBackend._evaluate().

    The LLM receives the full world state, the NL instruction, the action type,
    free slot names + types, and already-fixed slot values. It returns this schema
    with a resolved SlotValue for every free slot.

    Generic across all action types — no per-action subclassing needed.
    """

    action_type: str
    """The action class name being resolved (echoed back for validation)."""

    slots: List[SlotValue]
    """One entry per free slot in the Match expression."""

    overall_reasoning: str = ""
    """
    High-level explanation of the resolution strategy.
    E.g. "The milk is the only FoodItem on the table; left arm is free."
    """


class ActionClassification(BaseModel):
    """
    Output of the action classification step used by nl_plan() factory.

    The LLM is given the list of available action class names and picks the
    most appropriate one for the NL instruction.
    """

    action_type: str
    """
    Exact Python class name of the chosen action.
    E.g. "PickUpAction", "NavigateAction", "PlaceAction".
    Must match a key in the action registry.
    """

    confidence: float = 1.0
    """LLM's self-reported confidence (0.0–1.0). Informational only."""

    reasoning: str = ""
    """Why this action type was chosen."""
