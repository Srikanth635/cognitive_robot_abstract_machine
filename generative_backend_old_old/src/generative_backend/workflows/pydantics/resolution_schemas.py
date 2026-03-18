"""Phase 2 Pydantic schemas: LLM discrete-parameter resolution output.

These schemas receive world context + the known partial parameters and ask the
LLM to reason about and fill only the *discrete* unknowns.  All field values are
strictly typed to enum literals so that parsing is deterministic – no free text
that needs post-processing.

Design principle: one schema per action type, containing exactly the discrete
parameters that the LLM handles.  Continuous/spatial parameters (x, y position)
are *not* included here – they are delegated to the probabilistic backend.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
