"""Pydantic schemas for PlaceAction — slot filling (Phase 1) and discrete resolution (Phase 2).

All types that an LLM produces or consumes for a PlaceAction live here.

Phase 1 output (slot filling):
    ``PlaceSlotSchema`` — what the slot-filler LLM extracts from a NL instruction.
    Both ``object_description`` and ``target_description`` are always required.
    ``arm`` is optional; left ``None`` when not mentioned.

Phase 2 output (discrete resolution):
    ``PlaceDiscreteResolutionSchema`` — what the resolver LLM fills in after
    seeing the world context.  For PlaceAction, only arm selection is discrete;
    continuous placement pose is handled by the probabilistic backend.

``EntityDescriptionSchema`` is imported from ``pick_up`` to avoid duplication —
the held object is described identically for both action types.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from .pick_up import EntityDescriptionSchema


# ── PlaceAction Phase 1: slot filling ─────────────────────────────────────────


class TargetLocationDescriptionSchema(BaseModel):
    """Semantic description of a target placement location extracted from NL.

    Pre-grounding — captures what the LLM understood about where the object
    should be placed.  The ``EntityGrounder`` converts this into a concrete
    ``Body`` (surface, container, or spatial region).
    """

    name: str = Field(
        description="The surface or location name as mentioned in the instruction "
                    "(e.g. 'table', 'kitchen counter', 'shelf', 'left side')."
    )
    semantic_type: Optional[str] = Field(
        default=None,
        description="Ontological/semantic type hint if inferrable from context "
                    "(e.g. 'Table', 'Surface', 'Container').  Null if unknown.",
    )
    spatial_context: Optional[str] = Field(
        default=None,
        description="Spatial refinement of the target placement "
                    "(e.g. 'to the left of the mug', 'in the corner', 'on the right side').  "
                    "Null if not mentioned.",
    )
    attributes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional discriminating attributes of the target location "
                    "(e.g. {'color': 'wooden', 'size': 'small'}).  Null if none.",
    )


class PlaceSlotSchema(BaseModel):
    """Slot-filling output for a PlaceAction.

    Mirrors the parameters of ``pycram.robot_plans.actions.core.place.PlaceAction``
    at the leaf level.  Both entity descriptions are always required; ``arm`` is
    optional and left ``None`` when not mentioned.

    The ``action_type`` literal doubles as a Pydantic discriminator so that a
    ``Union[PickUpSlotSchema, PlaceSlotSchema]`` can be deserialised
    unambiguously from LLM output.
    """

    action_type: Literal["PlaceAction"] = "PlaceAction"

    object_description: EntityDescriptionSchema = Field(
        description="Semantic description of the object to place (currently held by robot).  "
                    "Always required."
    )
    target_description: TargetLocationDescriptionSchema = Field(
        description="Semantic description of where to place the object.  Always required."
    )
    arm: Optional[Literal["LEFT", "RIGHT"]] = Field(
        default=None,
        description="Which arm to use.  Null unless the instruction explicitly "
                    "names an arm (e.g. 'with your left arm', 'use the right arm').  "
                    "'BOTH' is not valid for PlaceAction.",
    )


# ── PlaceAction Phase 2: discrete resolution ──────────────────────────────────


class PlaceDiscreteResolutionSchema(BaseModel):
    """Resolved discrete parameters for PlaceAction.

    The LLM receives world context (the object currently held, target surface
    pose, robot configuration) and resolves which arm should perform the
    placement.  All values are strictly typed to valid enum literals.
    """

    arm: Literal["LEFT", "RIGHT"] = Field(
        description="Which arm the robot should use to place the object.  "
                    "Choose based on which arm is currently holding the object "
                    "and the target surface's position relative to the robot.  "
                    "If arm state is unknown, prefer the arm closer to the target."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the arm choice, referencing "
                    "the current arm state, target surface location, and robot "
                    "configuration."
    )
