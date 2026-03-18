"""Phase 1 Pydantic schemas: LLM slot-filling output for PickUpAction.

The LLM extracts what it *knows* from the NL instruction and leaves everything
else as ``None``.  These schemas mirror the leaf parameters of ``PickUpAction``
but carry only the information an LLM can reliably produce from free text:

- object_description  → entity grounding happens downstream (EntityGrounder)
- arm                 → Optional; filled only when the instruction is explicit
- grasp_params        → Optional; filled only when approach/orientation is mentioned

``None`` on any field is a deliberate signal: "this parameter is underspecified
and must be resolved in Phase 2."
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class EntityDescriptionSchema(BaseModel):
    """Semantic description of an object entity extracted from NL.

    This is pre-grounding – it describes what the LLM *understood* about the
    object.  The EntityGrounder converts this into a concrete ``Body``.
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


class GraspParamsSchema(BaseModel):
    """Grasp configuration parameters extractable from NL.

    Only ``approach_direction`` and ``vertical_alignment`` are discrete enum
    values that a LLM can meaningfully infer from free text.  ``rotate_gripper``
    is set only when the instruction explicitly requests rotation.

    Note: ``manipulator`` (the physical arm object) is NOT part of this schema –
    it is a runtime object injected by the hybrid resolver from the robot context.
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
    """Complete slot-filling output for a PickUpAction.

    Mirrors the parameters of ``pycram.robot_plans.actions.core.pick_up.PickUpAction``
    at the leaf level.  Fields that are not resolvable from NL alone are ``None``
    and will be filled by Phase 2 (hybrid resolver).
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
