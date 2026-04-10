"""Shared Pydantic schemas used across all action types.

Fully generic — no robot, pycram, or sdt references.
"""

from __future__ import annotations

from typing_extensions import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityDescriptionSchema(BaseModel):
    """Semantic description of a world entity to be grounded via SymbolGraph.

    The ``role`` field maps directly to the pycram action constructor parameter
    name this entity will fill (e.g. ``"object_designator"``, ``"target_location"``).
    The grounder uses the remaining fields to find the matching Symbol instance.
    """

    role: str = Field(
        description=(
            "The parameter name this entity fills in the action constructor "
            "(e.g. 'object_designator', 'target_location', 'region'). "
            "Must match the exact pycram action field name."
        )
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "The entity name or noun phrase as mentioned in the instruction "
            "(e.g. 'cup', 'red mug', 'milk bottle')."
        ),
    )
    semantic_type: Optional[str] = Field(
        default=None,
        description=(
            "Ontological/semantic type hint if inferrable "
            "(e.g. 'DrinkingContainer', 'HasSupportingSurface'). "
            "Must be an exact Symbol subclass name visible in the world context. "
            "Null if unknown."
        ),
    )
    spatial_context: Optional[str] = Field(
        default=None,
        description=(
            "Spatial relationship that can narrow down candidates "
            "(e.g. 'on the table', 'inside the fridge'). Null if not mentioned."
        ),
    )
    attributes: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Additional discriminating attributes "
            "(e.g. {'color': 'red', 'size': 'small'}). Null if none."
        ),
    )


class ActionSlotSchema(BaseModel):
    """Generic slot-filling output for any action type.

    - ``entities``   — world entities that need SymbolGraph grounding, each tagged
                       with a ``role`` matching the pycram constructor parameter name.
    - ``parameters`` — all non-entity parameters (enum choices, primitives, etc.)
                       keyed by the pycram field name.
    - ``manner``     — optional execution style hint ("carefully", "quickly").
    - ``constraints``— optional list of constraints ("without spilling it").
    """

    action_type: str = Field(
        description="Action type string matching a registered ActionHandler key."
    )
    entities: List[EntityDescriptionSchema] = Field(
        default_factory=list,
        description=(
            "All world entities involved in the action. Each must have a ``role`` "
            "matching the pycram action constructor parameter it fills."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Non-entity action parameters keyed by their pycram field name "
            "(e.g. arm, approach_direction, torso_state, keep_joint_states)."
        ),
    )
    manner: Optional[str] = Field(
        default=None,
        description="Execution style hint from the instruction ('carefully', 'slowly'). Null if not mentioned.",
    )
    constraints: Optional[List[str]] = Field(
        default=None,
        description="Explicit constraints from the instruction ('without spilling'). Null if none.",
    )
