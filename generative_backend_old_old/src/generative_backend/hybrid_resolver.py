"""Phase 2: PartialDesignator[PickUpAction] → PickUpAction (hybrid resolution).

Responsibility split:
  Discrete parameters (arm, approach_direction, vertical_alignment, rotate_gripper)
      → LLM resolver (``DiscreteResolverAgent``)
  Continuous parameters (standing position x, y for MoveAndPickUpAction)
      → Probabilistic backend (``MoveAndPickUpParameterizer``)

For ``PickUpAction`` specifically, all parameters are discrete once the object is
grounded, so the probabilistic backend is *not* invoked.  The hook is present so
that upgrading to ``MoveAndPickUpAction`` simply means switching the resolver class.

## World context construction

The LLM resolver needs a concise world snapshot that lets it reason about:
  - Where is the object relative to the robot?
  - What semantic annotations does the object have?
  - Are there obvious spatial constraints?

``WorldContextBuilder`` serialises this from the SDT world into a human-readable
string that is sent as part of the prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.robot_plans.actions.core.pick_up import PickUpAction

from .partial_designator_builder import WorldContext
from .workflows.agents.discrete_resolver import run_discrete_resolver
from .workflows.pydantics.resolution_schemas import PickUpDiscreteResolutionSchema

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _pose_to_xyz(pose):
    """Return (x, y, z) floats from a HomogeneousTransformationMatrix, or None."""
    try:
        pt = pose.to_position()
        return float(pt.x), float(pt.y), float(pt.z)
    except Exception:
        return None


# ── World context builder ──────────────────────────────────────────────────────


class WorldContextBuilder:
    """Builds a human-readable world snapshot for the LLM resolver."""

    def __init__(self, world: World) -> None:
        self._world = world

    def build_for_pickup(
        self,
        partial: PartialDesignator[PickUpAction],
    ) -> str:
        """Return a concise world context string relevant to the pick-up task."""
        lines: List[str] = []

        # ── Robot pose ─────────────────────────────────────────────────────────
        robot_xyz = None
        try:
            robot = self._world.get_semantic_annotations_by_type(AbstractRobot)[0]
            # AbstractRobot.base is a Base (KinematicChain); .root is its root Body
            robot_body = robot.base.root if (robot.base is not None) else None
            if robot_body is not None:
                robot_xyz = _pose_to_xyz(robot_body.global_pose)
                if robot_xyz:
                    rx, ry, rz = robot_xyz
                    lines.append(f"Robot position: x={rx:.3f}, y={ry:.3f}, z={rz:.3f}")
        except Exception:
            lines.append("Robot position: unknown")

        # ── Object(s) info ─────────────────────────────────────────────────────
        object_param = partial.kwargs.get("object_designator")
        objects: List[Body] = (
            object_param if isinstance(object_param, list) else [object_param]
        ) if object_param is not None else []

        for obj in objects:
            name = str(getattr(obj, "name", obj))
            obj_xyz = None
            try:
                obj_xyz = _pose_to_xyz(obj.global_pose)
                if obj_xyz:
                    ox, oy, oz = obj_xyz
                    lines.append(f"Object '{name}': x={ox:.3f}, y={oy:.3f}, z={oz:.3f}")

                    # Relative position vs robot (robot_xyz already computed above)
                    if robot_xyz is not None:
                        rx, ry, rz = robot_xyz
                        dx, dy, dz = ox - rx, oy - ry, oz - rz
                        side = "right" if dy < 0 else "left"
                        front_back = "in front of" if dx > 0 else "behind"
                        lines.append(
                            f"  → Object is {abs(dx):.2f}m {front_back} and "
                            f"{abs(dy):.2f}m to the {side} of the robot, "
                            f"{abs(dz):.2f}m {'above' if dz > 0 else 'below'} robot origin."
                        )
                else:
                    lines.append(f"Object '{name}': pose unknown")
            except Exception:
                lines.append(f"Object '{name}': pose unknown")

            # Semantic annotations
            try:
                annotations = self._world.get_semantic_annotations_of_body(obj)
                ann_types = [type(a).__name__ for a in annotations]
                if ann_types:
                    lines.append(f"  → Semantic types: {', '.join(ann_types)}")
            except Exception:
                pass

            # Bounding box (rough size)
            try:
                bb = obj.collision.as_bounding_box_collection_in_frame(obj).bounding_box()
                dims = bb.dimensions
                lines.append(
                    f"  → Bounding box (w×d×h): "
                    f"{dims[0]:.3f} × {dims[1]:.3f} × {dims[2]:.3f} m"
                )
            except Exception:
                pass

        return "\n".join(lines) if lines else "World context unavailable."


# ── Parameter introspection helpers ───────────────────────────────────────────


def _known_params_summary(partial: PartialDesignator[PickUpAction]) -> str:
    """Return a human-readable summary of already-specified parameters."""
    lines = []
    arm = partial.kwargs.get("arm")
    if arm is not None:
        lines.append(f"arm = {arm}")
    grasp = partial.kwargs.get("grasp_description")
    if grasp is not None:
        lines.append(
            f"approach_direction = {grasp.approach_direction.name}, "
            f"vertical_alignment = {grasp.vertical_alignment.name}, "
            f"rotate_gripper = {grasp.rotate_gripper}"
        )
    return "\n".join(lines) if lines else "None – all discrete parameters are unspecified."


def _missing_params_description(partial: PartialDesignator[PickUpAction]) -> str:
    """Return a description of the parameters that still need to be resolved."""
    missing = []
    if partial.kwargs.get("arm") is None:
        missing.append("arm  (choose LEFT or RIGHT based on object position)")
    if partial.kwargs.get("grasp_description") is None:
        missing.extend([
            "approach_direction  (FRONT / BACK / LEFT / RIGHT)",
            "vertical_alignment  (TOP / BOTTOM / NoAlignment)",
            "rotate_gripper      (true / false)",
        ])
    return "\n".join(missing) if missing else "All parameters already specified."


# ── Hybrid resolver ────────────────────────────────────────────────────────────


@dataclass
class HybridPickUpResolver:
    """Resolves a ``PartialDesignator[PickUpAction]`` to a full ``PickUpAction``.

    Resolution strategy:
      1. If no parameters are missing → call ``partial.resolve()`` immediately.
      2. Build a world context string for the LLM.
      3. Call the LLM discrete resolver to fill arm / grasp parameters.
      4. Reconstruct ``GraspDescription`` with the resolved values + manipulator.
      5. Return the fully specified ``PickUpAction``.

    :param world: SDT world instance.
    :param world_context: Contains the robot manipulator for GraspDescription.
    """

    world: World
    world_context: WorldContext

    def resolve(
        self,
        partial: PartialDesignator[PickUpAction],
    ) -> PickUpAction:
        """Resolve *partial* to a fully specified ``PickUpAction``.

        :param partial: Partially specified designator from Phase 1.
        :return: A ``PickUpAction`` with all parameters set.
        :raises RuntimeError: When LLM resolution fails and no fallback is available.
        """
        # ── Fast path: already fully specified ───────────────────────────────
        if not partial.missing_parameter():
            logger.debug("PartialDesignator fully specified – resolving directly.")
            return partial.resolve()

        # ── Build world context for LLM ───────────────────────────────────────
        ctx_builder = WorldContextBuilder(self.world)
        world_ctx_str = ctx_builder.build_for_pickup(partial)
        known_str = _known_params_summary(partial)
        missing_str = _missing_params_description(partial)

        logger.debug("Calling discrete resolver LLM.\nMissing: %s", missing_str)

        # ── LLM discrete resolution ───────────────────────────────────────────
        resolution: Optional[PickUpDiscreteResolutionSchema] = run_discrete_resolver(
            world_context=world_ctx_str,
            known_parameters=known_str,
            parameters_to_resolve=missing_str,
        )

        if resolution is None:
            raise RuntimeError(
                "Discrete resolver LLM returned None.  "
                "Check logs for the underlying error."
            )

        logger.debug("LLM reasoning: %s", resolution.reasoning)

        # ── Merge resolved values into partial ────────────────────────────────
        return self._build_action(partial, resolution)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_action(
        self,
        partial: PartialDesignator[PickUpAction],
        resolution: PickUpDiscreteResolutionSchema,
    ) -> PickUpAction:
        """Construct a ``PickUpAction`` by merging partial + resolved values."""

        # arm: prefer the one already in partial, fall back to LLM resolution
        arm: Arms = partial.kwargs.get("arm") or Arms[resolution.arm]

        # grasp_description: prefer partial's if present
        grasp_description: Optional[GraspDescription] = partial.kwargs.get(
            "grasp_description"
        )
        if grasp_description is None:
            grasp_description = self._build_grasp_description(arm, resolution)

        # object_designator: take the first grounded body
        object_designator = partial.kwargs.get("object_designator")
        if isinstance(object_designator, list):
            object_designator = object_designator[0]

        return PickUpAction(
            object_designator=object_designator,
            arm=arm,
            grasp_description=grasp_description,
        )

    def _build_grasp_description(
        self,
        arm: Arms,
        resolution: PickUpDiscreteResolutionSchema,
    ) -> GraspDescription:
        """Construct ``GraspDescription`` from resolved values + injected manipulator."""
        manipulator: Optional[Manipulator] = self.world_context.manipulator

        if manipulator is None:
            raise RuntimeError(
                "Cannot construct GraspDescription: manipulator is None in "
                "WorldContext.  Provide the Manipulator from the robot arm chain."
            )

        return GraspDescription(
            approach_direction=ApproachDirection[resolution.approach_direction],
            vertical_alignment=VerticalAlignment[resolution.vertical_alignment],
            rotate_gripper=resolution.rotate_gripper,
            manipulator=manipulator,
        )
