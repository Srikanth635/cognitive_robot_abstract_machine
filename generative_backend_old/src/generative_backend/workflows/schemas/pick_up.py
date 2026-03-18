"""Pydantic schemas for PickUpAction — slot filling (Phase 1) and discrete resolution (Phase 2).

All types that an LLM produces or consumes for a PickUpAction live here.

Phase 1 output (slot filling):
    ``PickUpSlotSchema`` — what the slot-filler LLM extracts from a NL instruction.
    Fields left ``None`` are free variables to be resolved in Phase 2.

Phase 2 output (discrete resolution):
    ``PickUpDiscreteResolutionSchema`` — what the resolver LLM fills in after
    seeing the world context.  All values are strictly typed enum literals so
    that downstream parsing is deterministic.

Shared sub-schemas:
    ``EntityDescriptionSchema`` — pre-grounding semantic description of any entity.
    ``GraspParamsSchema``       — optional grasp configuration from the instruction.

``EntityDescriptionSchema`` is also imported by ``place.py`` to avoid duplication,
as the object-to-place is described in the same way as the object-to-pick-up.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


# ── Shared entity description ──────────────────────────────────────────────────


class EntityDescriptionSchema(BaseModel):
    """Semantic description of an object entity extracted from NL.

    This is pre-grounding – it captures what the LLM *understood* about the
    object.  The ``EntityGrounder`` converts this into a concrete ``Body``.
    """

    name: str = Field(
        description="The object name or noun phrase as mentioned in the instruction "
                    "(e.g. 'cup', 'red mug', 'milk bottle')."
    )
    semantic_type: Optional[str] = Field(
        default=None,
        description="Ontological/semantic type hint if inferrable from context "
                    "(e.g. 'Artifact', 'Container', 'FoodItem').  Null if unknown.",
    )
    spatial_context: Optional[str] = Field(
        default=None,
        description="Spatial relationship that can narrow down candidates "
                    "(e.g. 'on the table', 'inside the fridge', 'to the left').  "
                    "Null if not mentioned.",
    )
    attributes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional discriminating attributes from the instruction "
                    "(e.g. {'color': 'red', 'size': 'small'}).  Null if none.",
    )


# ── PickUp Phase 1: slot filling ───────────────────────────────────────────────


class GraspParamsSchema(BaseModel):
    """Grasp configuration parameters extractable from NL.

    Only the three discrete values are modelled here — they are the ones an
    LLM can reliably infer from free text.  The physical ``manipulator`` object
    is a runtime artefact injected later from the robot context.
    """

    approach_direction: Optional[Literal["FRONT", "BACK", "LEFT", "RIGHT"]] = Field(
        default=None,
        description="Direction from which to approach the object.  "
                    "Null unless the instruction specifies an approach side.",
    )
    vertical_alignment: Optional[Literal["TOP", "BOTTOM", "NoAlignment"]] = Field(
        default=None,
        description="Vertical alignment of the gripper.  "
                    "Null unless explicitly mentioned.",
    )
    rotate_gripper: Optional[bool] = Field(
        default=None,
        description="Whether to rotate the gripper 90°.  "
                    "Null unless the instruction mentions rotation.",
    )


class PickUpSlotSchema(BaseModel):
    """Slot-filling output for a PickUpAction.

    Mirrors the parameters of ``pycram.robot_plans.actions.core.pick_up.PickUpAction``
    at the leaf level.  Fields that are not resolvable from NL alone are ``None``
    and will be filled by Phase 2 (hybrid resolver).

    The ``action_type`` literal doubles as a Pydantic discriminator so that a
    ``Union[PickUpSlotSchema, PlaceSlotSchema]`` can be deserialised
    unambiguously from LLM output.
    """

    action_type: Literal["PickUpAction"] = "PickUpAction"

    object_description: EntityDescriptionSchema = Field(
        description="Semantic description of the object to pick up.  Always required."
    )
    arm: Optional[Literal["LEFT", "RIGHT", "BOTH"]] = Field(
        default=None,
        description="Which arm to use.  Null unless the instruction explicitly "
                    "names an arm (e.g. 'with your left arm', 'use the right arm').",
    )
    grasp_params: Optional[GraspParamsSchema] = Field(
        default=None,
        description="Grasp configuration.  Null unless the instruction mentions "
                    "approach direction, orientation, or gripper rotation.",
    )


# ── PickUp Phase 2: discrete resolution ───────────────────────────────────────


class PickUpDiscreteResolutionSchema(BaseModel):
    """Resolved discrete parameters for PickUpAction.

    The LLM receives world context (object geometry, robot pose, semantic
    annotations) and fills the three grasp parameters plus arm selection.
    All values are strictly typed to valid enum literals.
    """

    arm: Literal["LEFT", "RIGHT"] = Field(
        description="Which arm the robot should use.  Choose based on the object's "
                    "position relative to the robot (e.g. object to the right → RIGHT arm)."
    )
    approach_direction: Literal["FRONT", "BACK", "LEFT", "RIGHT"] = Field(
        description="Direction from which the gripper approaches the object.  "
                    "FRONT: along the robot's forward axis.  "
                    "BACK: from behind the object.  "
                    "LEFT/RIGHT: from the lateral sides."
    )
    vertical_alignment: Literal["TOP", "BOTTOM", "NoAlignment"] = Field(
        description="Vertical gripper alignment.  "
                    "TOP: gripper comes from above.  "
                    "BOTTOM: from below.  "
                    "NoAlignment: purely lateral, no vertical bias."
    )
    rotate_gripper: bool = Field(
        description="Whether to rotate the gripper 90° around its approach axis.  "
                    "True for elongated objects whose longest axis is perpendicular "
                    "to the default gripper orientation."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the choices made, referencing "
                    "the object's pose and the robot's configuration."
    )
