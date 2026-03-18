"""Phase 1 output → PartialDesignator[PickUpAction].

This module converts the LLM-produced ``PickUpSlotSchema`` plus the grounded
``Body`` objects into a ``PartialDesignator[PickUpAction]`` that the rest of
the pycram pipeline can consume unchanged.

Key decisions made here:
- If ``arm`` is null → ``arm=None`` in the PartialDesignator (free variable).
- If ``grasp_params`` is fully specified → construct a ``GraspDescription`` and
  set it directly.  If partially specified or null → ``grasp_description=None``
  (free variable for Phase 2 to resolve).
- ``GraspDescription.manipulator`` is a runtime robot object that cannot come
  from the LLM.  It is therefore injected from the ``WorldContext`` here only
  when a fully specified grasp is available; otherwise the manipulator is also
  left for Phase 2.
- Multiple candidate bodies from the grounder become a *list* in
  ``object_designator`` – pycram's ``PartialDesignator`` already handles lists
  as parameter domains via ``generate_permutations()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.robot_plans.actions.core.pick_up import PickUpAction

from .workflows.pydantics.pick_up_schemas import GraspParamsSchema, PickUpSlotSchema

logger = logging.getLogger(__name__)


# ── Public dataclass for context injection ─────────────────────────────────────


@dataclass
class WorldContext:
    """Runtime objects needed to finalise a ``PartialDesignator``.

    The ``manipulator`` is the semantic annotation on the robot arm that the
    ``GraspDescription`` requires.  It is obtained from the SDT world at
    pipeline initialisation time and injected here so the builder does not
    need to re-query the world.

    If ``manipulator`` is ``None`` the builder cannot construct a full
    ``GraspDescription`` even when all grasp parameters are specified by the LLM;
    in that case ``grasp_description`` remains ``None`` and Phase 2 must handle it.
    """

    manipulator: Optional[Manipulator] = None
    """The robot manipulator (arm end-effector) to embed in GraspDescription."""


# ── Public API ─────────────────────────────────────────────────────────────────


def build_partial_designator(
    slot_schema: PickUpSlotSchema,
    grounded_bodies: List[Body],
    world_context: Optional[WorldContext] = None,
) -> PartialDesignator[PickUpAction]:
    """Convert a slot schema + grounded bodies into a PartialDesignator.

    Parameters left as ``None`` are *free variables* – they will be resolved by
    the hybrid resolver in Phase 2.

    :param slot_schema: Output of the Phase 1 slot-filler LLM.
    :param grounded_bodies: One or more ``Body`` objects resolved by the grounder.
    :param world_context: Runtime context carrying the robot manipulator.
    :return: ``PartialDesignator[PickUpAction]`` ready for Phase 2.
    :raises ValueError: When ``grounded_bodies`` is empty.
    """
    if not grounded_bodies:
        raise ValueError(
            "Cannot build PartialDesignator: grounded_bodies is empty.  "
            "Ensure EntityGrounder resolved at least one Body before calling "
            "build_partial_designator."
        )

    # ── arm ────────────────────────────────────────────────────────────────────
    arm: Optional[Arms] = _parse_arm(slot_schema.arm)

    # ── grasp_description ──────────────────────────────────────────────────────
    grasp_description: Optional[GraspDescription] = _parse_grasp_description(
        slot_schema.grasp_params,
        world_context,
    )

    # ── object_designator ─────────────────────────────────────────────────────
    # Use the list directly if multiple candidates (PartialDesignator handles
    # iteration over the domain).  Use the single Body if unambiguous.
    object_designator = grounded_bodies if len(grounded_bodies) > 1 else grounded_bodies[0]

    logger.debug(
        "PartialDesignator built – object=%s, arm=%s, grasp=%s",
        [str(getattr(b, "name", b)) for b in grounded_bodies],
        arm,
        grasp_description,
    )

    return PartialDesignator(
        PickUpAction,
        object_designator=object_designator,
        arm=arm,
        grasp_description=grasp_description,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────


def _parse_arm(arm_str: Optional[str]) -> Optional[Arms]:
    """Convert the LLM's arm string literal to ``Arms`` enum, or ``None``."""
    if arm_str is None:
        return None
    try:
        return Arms[arm_str]
    except KeyError:
        logger.warning("Unknown arm value '%s' from LLM – setting to None.", arm_str)
        return None


def _parse_grasp_description(
    grasp_params: Optional[GraspParamsSchema],
    world_context: Optional[WorldContext],
) -> Optional[GraspDescription]:
    """Attempt to construct a full ``GraspDescription`` from LLM grasp params.

    Returns ``None`` (free variable) in three cases:
    1. ``grasp_params`` is entirely null.
    2. ``approach_direction`` is null (minimum required field).
    3. ``manipulator`` is not available in ``world_context``.

    A partial ``GraspParamsSchema`` where some sub-fields are null (e.g. only
    ``approach_direction`` is set but not ``vertical_alignment``) also returns
    ``None``, delegating the complete resolution to Phase 2.
    """
    if grasp_params is None:
        return None

    # All three discriminating fields must be present to build a complete desc.
    if (
        grasp_params.approach_direction is None
        or grasp_params.vertical_alignment is None
        or grasp_params.rotate_gripper is None
    ):
        logger.debug(
            "Grasp params partially specified (%s) – deferring to Phase 2.",
            grasp_params,
        )
        return None

    manipulator = world_context.manipulator if world_context else None
    if manipulator is None:
        logger.debug(
            "Manipulator not available in WorldContext – deferring grasp to Phase 2."
        )
        return None

    try:
        approach = ApproachDirection[grasp_params.approach_direction]
        vertical = VerticalAlignment[grasp_params.vertical_alignment]
    except KeyError as exc:
        logger.warning("Could not parse grasp enum value: %s", exc)
        return None

    return GraspDescription(
        approach_direction=approach,
        vertical_alignment=vertical,
        rotate_gripper=grasp_params.rotate_gripper,
        manipulator=manipulator,
    )
