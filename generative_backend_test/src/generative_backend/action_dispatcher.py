"""Action dispatcher: routes a typed slot schema to the correct handler.

This module is the single place that knows how to execute any supported action
type.  Each action type has an ``ActionHandler`` that encapsulates every step
from slot schema to concrete pycram action:

  - How many entities to ground (and which descriptions to use).
  - How to build a ``PartialDesignator`` for that action.
  - Which discrete parameters to resolve via LLM and how.
  - How to assemble the final pycram action object.

``WorldContext`` is also defined here as it is shared by the dispatcher and
the pipeline entry point (``action_pipeline.py``).

## Adding a new action type

1. Implement ``YourActionHandler(ActionHandler)`` in this file.
2. Register it: ``ActionDispatcher.register("YourAction", YourActionHandler)``.
3. Add ``YourActionSlotSchema`` to ``workflows/schemas/`` and a prompt to
   ``workflows/prompts/``.
4. Update ``workflows/agents/slot_filler.py`` to classify and return the new
   schema.

The pipeline (``action_pipeline.py``) requires zero changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction

from .entity_grounder import EntityGrounder, GroundingResult
from .workflows._utils import _pose_to_xyz
from .workflows.nodes.resolver import run_pickup_resolver
from .workflows.schemas.common import EntityDescriptionSchema
from .workflows.schemas.pick_up import (
    GraspParamsSchema,
    PickUpDiscreteResolutionSchema,
    PickUpSlotSchema,
)
from .workflows.schemas.place import PlaceDiscreteResolutionSchema, PlaceSlotSchema
from .workflows.nodes.slot_filler import ActionSlotSchema

logger = logging.getLogger(__name__)


# ── Runtime context ────────────────────────────────────────────────────────────


@dataclass
class WorldContext:
    """Runtime objects shared across all action handlers.

    ``manipulator`` is the semantic annotation on the robot arm required by
    ``GraspDescription`` (PickUpAction).  It is obtained once at pipeline
    initialisation and injected here so handlers do not re-query the world.
    """

    manipulator: Optional[Manipulator] = None


# ── ActionHandler base ─────────────────────────────────────────────────────────


class ActionHandler(ABC):
    """Base class for action-specific execution handlers."""

    def __init__(self, world: World, world_context: WorldContext) -> None:
        self._world = world
        self._world_context = world_context
        self._grounder = EntityGrounder(world)

    @abstractmethod
    def execute(self, schema: Any) -> Any:
        """Ground entities, build partial designator, resolve, return action."""

    def _ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        result = self._grounder.ground(description)
        if result.warning:
            logger.warning("Grounding warning: %s", result.warning)
        return result

    def _get_robot_context(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], List[str]]:
        """Return robot (x, y, z) and initial context lines including robot position.

        :return: Tuple of (robot_xyz_or_None, lines_list).  ``lines_list`` already
                 contains the robot position string (or 'unknown') as its first entry.
        """
        lines: List[str] = []
        robot_xyz: Optional[Tuple[float, float, float]] = None
        try:
            robot = self._world.get_semantic_annotations_by_type(AbstractRobot)[0]
            robot_body = robot.base.root if robot.base is not None else None
            if robot_body is not None:
                robot_xyz = _pose_to_xyz(robot_body.global_pose)
                if robot_xyz:
                    rx, ry, rz = robot_xyz
                    lines.append(f"Robot position: x={rx:.3f}, y={ry:.3f}, z={rz:.3f}")
        except Exception:
            lines.append("Robot position: unknown")
        return robot_xyz, lines


# ── PickUp handler ─────────────────────────────────────────────────────────────


class PickUpActionHandler(ActionHandler):
    """Handles the full execution chain for PickUpAction.

    Execution steps (all contained here):
      1. Ground the object.
      2. Build a ``PartialDesignator[PickUpAction]`` from the slot schema.
      3. Resolve free parameters (arm, grasp) via world context + LLM.
      4. Return the fully specified ``PickUpAction``.
    """

    _resolver_llm: Any = None  # lazy-initialised at class level

    @classmethod
    def _get_resolver_llm(cls) -> Any:
        if cls._resolver_llm is None:
            from .workflows.llm_configuration import default_llm
            cls._resolver_llm = default_llm.with_structured_output(
                PickUpDiscreteResolutionSchema, method="function_calling"
            )
        return cls._resolver_llm

    def execute(self, schema: PickUpSlotSchema) -> PickUpAction:
        # Step 1 — ground the object
        grounding = self._ground(schema.object_description)
        if not grounding.bodies:
            raise RuntimeError(
                f"No Body found for object '{schema.object_description.name}'. "
                "Check that the object exists in the world."
            )
        logger.info(
            "PickUp grounding – %d body/bodies (tier=%s).",
            len(grounding.bodies),
            grounding.tier,
        )

        # Step 2 — build PartialDesignator
        partial = self._build_partial(schema, grounding.bodies)

        # Step 3 — resolve free parameters
        if not partial.missing_parameter():
            logger.debug("PartialDesignator fully specified – resolving directly.")
            return partial.resolve()

        resolution = self._resolve_discrete(partial)
        logger.debug("LLM reasoning: %s", resolution.reasoning)

        # Step 4 — assemble action
        return self._build_action(partial, resolution)

    # ── PartialDesignator construction ─────────────────────────────────────────

    def _build_partial(
        self,
        schema: PickUpSlotSchema,
        grounded_bodies: List[Body],
    ) -> PartialDesignator[PickUpAction]:
        arm = self._parse_arm(schema.arm)
        grasp = self._parse_grasp(schema.grasp_params)
        obj = grounded_bodies if len(grounded_bodies) > 1 else grounded_bodies[0]
        return PartialDesignator(
            PickUpAction,
            object_designator=obj,
            arm=arm,
            grasp_description=grasp,
        )

    def _parse_arm(self, arm_str: Optional[str]) -> Optional[Arms]:
        if arm_str is None:
            return None
        try:
            return Arms[arm_str]
        except KeyError:
            logger.warning("Unknown arm value '%s' from LLM – ignoring.", arm_str)
            return None

    def _parse_grasp(
        self,
        params: Optional[GraspParamsSchema],
    ) -> Optional[GraspDescription]:
        if params is None:
            return None
        if (
            params.approach_direction is None
            or params.vertical_alignment is None
            or params.rotate_gripper is None
        ):
            logger.debug("Grasp params partially specified – deferring to LLM resolver.")
            return None
        manipulator = self._world_context.manipulator
        if manipulator is None:
            logger.debug("Manipulator unavailable – deferring grasp to LLM resolver.")
            return None
        try:
            return GraspDescription(
                approach_direction=ApproachDirection[params.approach_direction],
                vertical_alignment=VerticalAlignment[params.vertical_alignment],
                rotate_gripper=params.rotate_gripper,
                manipulator=manipulator,
            )
        except KeyError as exc:
            logger.warning("Could not parse grasp enum: %s", exc)
            return None

    # ── Discrete resolution ────────────────────────────────────────────────────

    def _resolve_discrete(
        self,
        partial: PartialDesignator[PickUpAction],
    ) -> PickUpDiscreteResolutionSchema:
        world_ctx = self._build_world_context(partial)
        known = self._known_params(partial)
        missing = self._missing_params(partial)
        logger.debug("PickUp world context:\n%s", world_ctx)
        logger.debug("PickUp known parameters: %s", known)
        logger.debug("PickUp missing parameters: %s", missing)

        resolution = run_pickup_resolver(
            world_context=world_ctx,
            known_parameters=known,
            parameters_to_resolve=missing,
        )
        if resolution is None:
            raise RuntimeError("Discrete resolver LLM returned None. Check logs.")
        return resolution

    def _build_world_context(self, partial: PartialDesignator[PickUpAction]) -> str:
        robot_xyz, lines = self._get_robot_context()

        object_param = partial.kwargs.get("object_designator")
        objects: List[Body] = (
            object_param if isinstance(object_param, list) else [object_param]
        ) if object_param is not None else []

        for obj in objects:
            name = str(getattr(obj, "name", obj))
            try:
                xyz = _pose_to_xyz(obj.global_pose)
                if xyz:
                    ox, oy, oz = xyz
                    lines.append(f"Object '{name}': x={ox:.3f}, y={oy:.3f}, z={oz:.3f}")
                    if robot_xyz:
                        rx, ry, rz = robot_xyz
                        dx, dy, dz = ox - rx, oy - ry, oz - rz
                        side = "right" if dy < 0 else "left"
                        front_back = "in front of" if dx > 0 else "behind"
                        lines.append(
                            f"  → {abs(dx):.2f}m {front_back} and "
                            f"{abs(dy):.2f}m to the {side} of the robot, "
                            f"{abs(dz):.2f}m {'above' if dz > 0 else 'below'} robot origin."
                        )
                else:
                    lines.append(f"Object '{name}': pose unknown")
            except Exception:
                lines.append(f"Object '{name}': pose unknown")

            try:
                ann_types = [
                    type(a).__name__
                    for a in self._world.get_semantic_annotations_of_body(obj)
                ]
                if ann_types:
                    lines.append(f"  → Semantic types: {', '.join(ann_types)}")
            except Exception:
                pass

            try:
                bb = obj.collision.as_bounding_box_collection_in_frame(obj).bounding_box()
                d = bb.dimensions
                lines.append(
                    f"  → Bounding box (w×d×h): {d[0]:.3f} × {d[1]:.3f} × {d[2]:.3f} m"
                )
            except Exception:
                pass

        return "\n".join(lines) if lines else "World context unavailable."

    @staticmethod
    def _known_params(partial: PartialDesignator[PickUpAction]) -> str:
        lines = []
        if partial.kwargs.get("arm") is not None:
            lines.append(f"arm = {partial.kwargs['arm']}")
        grasp = partial.kwargs.get("grasp_description")
        if grasp is not None:
            lines.append(
                f"approach_direction = {grasp.approach_direction.name}, "
                f"vertical_alignment = {grasp.vertical_alignment.name}, "
                f"rotate_gripper = {grasp.rotate_gripper}"
            )
        return "\n".join(lines) if lines else "None – all discrete parameters are unspecified."

    @staticmethod
    def _missing_params(partial: PartialDesignator[PickUpAction]) -> str:
        missing = []
        if partial.kwargs.get("arm") is None:
            missing.append("arm  (choose LEFT or RIGHT based on object position)")
        if partial.kwargs.get("grasp_description") is None:
            missing += [
                "approach_direction  (FRONT / BACK / LEFT / RIGHT)",
                "vertical_alignment  (TOP / BOTTOM / NoAlignment)",
                "rotate_gripper      (true / false)",
            ]
        return "\n".join(missing) if missing else "All parameters already specified."

    # ── Final action assembly ──────────────────────────────────────────────────

    def _build_action(
        self,
        partial: PartialDesignator[PickUpAction],
        resolution: PickUpDiscreteResolutionSchema,
    ) -> PickUpAction:
        arm: Arms = partial.kwargs.get("arm") or Arms[resolution.arm]

        grasp: Optional[GraspDescription] = partial.kwargs.get("grasp_description")
        if grasp is None:
            manipulator = self._world_context.manipulator
            if manipulator is None:
                raise RuntimeError(
                    "Cannot construct GraspDescription: manipulator is None in WorldContext."
                )
            grasp = GraspDescription(
                approach_direction=ApproachDirection[resolution.approach_direction],
                vertical_alignment=VerticalAlignment[resolution.vertical_alignment],
                rotate_gripper=resolution.rotate_gripper,
                manipulator=manipulator,
            )

        obj = partial.kwargs.get("object_designator")
        if isinstance(obj, list):
            obj = obj[0]

        return PickUpAction(object_designator=obj, arm=arm, grasp_description=grasp)


# ── Place handler ──────────────────────────────────────────────────────────────


class PlaceActionHandler(ActionHandler):
    """Handles the full execution chain for PlaceAction.

    Execution steps (all contained here):
      1. Ground the object being placed.
      2. Ground the target surface.
      3. Resolve arm (from schema or LLM).
      4. Return the fully specified ``PlaceAction``.
    """

    _resolver_llm: Any = None  # lazy-initialised at class level

    @classmethod
    def _get_resolver_llm(cls) -> Any:
        if cls._resolver_llm is None:
            from .workflows.llm_configuration import default_llm
            cls._resolver_llm = default_llm.with_structured_output(
                PlaceDiscreteResolutionSchema, method="function_calling"
            )
        return cls._resolver_llm

    def execute(self, schema: PlaceSlotSchema) -> PlaceAction:
        obj_grounding = self._ground(schema.object_description)
        if not obj_grounding.bodies:
            raise RuntimeError(
                f"No Body found for object '{schema.object_description.name}'. "
                "Check that the object exists in the world."
            )

        tgt_grounding = self._ground(schema.target_description)
        if not tgt_grounding.bodies:
            raise RuntimeError(
                f"No Body found for target '{schema.target_description.name}'. "
                "Check that the surface/location exists in the world."
            )

        logger.info(
            "Place grounding – object: %d body/bodies (tier=%s), "
            "target: %d body/bodies (tier=%s).",
            len(obj_grounding.bodies),
            obj_grounding.tier,
            len(tgt_grounding.bodies),
            tgt_grounding.tier,
        )

        obj_body = obj_grounding.bodies[0]
        tgt_body = tgt_grounding.bodies[0]

        arm: Optional[Arms] = Arms[schema.arm] if schema.arm else None
        partial: PartialDesignator[PlaceAction] = PartialDesignator(
            PlaceAction, object_designator=obj_body, arm=arm, target_location=tgt_body,
        )

        if partial.missing_parameter():
            resolution = self._resolve_arm(obj_grounding.bodies, tgt_grounding.bodies)
            arm = partial.kwargs.get("arm") or Arms[resolution.arm]
            logger.debug(
                "Place arm resolved by LLM: %s | %s", arm, resolution.reasoning
            )

        return PlaceAction(object_designator=obj_body, arm=arm, target_location=tgt_body)

    def _resolve_arm(
        self, obj_bodies: List[Body], tgt_bodies: List[Body]
    ) -> PlaceDiscreteResolutionSchema:
        from .workflows.nodes.resolver import run_place_resolver

        world_ctx = self._build_world_context(obj_bodies, tgt_bodies)
        resolution = run_place_resolver(
            world_context=world_ctx,
            known_parameters="None – arm not specified.",
            parameters_to_resolve=(
                "arm  (choose LEFT or RIGHT based on "
                "which arm holds the object and target position)"
            ),
        )
        if resolution is None:
            raise RuntimeError("Place resolver LLM returned None. Check logs.")
        return resolution

    def _build_world_context(
        self, obj_bodies: List[Body], tgt_bodies: List[Body]
    ) -> str:
        robot_xyz, lines = self._get_robot_context()

        for obj in obj_bodies:
            name = str(getattr(obj, "name", obj))
            try:
                xyz = _pose_to_xyz(obj.global_pose)
                lines.append(
                    f"Object to place '{name}': x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}"
                    if xyz
                    else f"Object to place '{name}': pose unknown"
                )
            except Exception:
                lines.append(f"Object to place '{name}': pose unknown")

        for tgt in tgt_bodies:
            name = str(getattr(tgt, "name", tgt))
            try:
                tgt_xyz = _pose_to_xyz(tgt.global_pose)
                if tgt_xyz:
                    tx, ty, tz = tgt_xyz
                    lines.append(
                        f"Target surface '{name}': x={tx:.3f}, y={ty:.3f}, z={tz:.3f}"
                    )
                    if robot_xyz is not None:
                        rx, ry, _ = robot_xyz
                        dx, dy = tx - rx, ty - ry
                        side = "right" if dy < 0 else "left"
                        front_back = "in front of" if dx > 0 else "behind"
                        lines.append(
                            f"  → {abs(dx):.2f}m {front_back} and "
                            f"{abs(dy):.2f}m to the {side} of the robot."
                        )
                else:
                    lines.append(f"Target surface '{name}': pose unknown")
            except Exception:
                lines.append(f"Target surface '{name}': pose unknown")

            try:
                types = [
                    type(a).__name__
                    for a in self._world.get_semantic_annotations_of_body(tgt)
                ]
                if types:
                    lines.append(f"  → Semantic types: {', '.join(types)}")
            except Exception:
                pass

        return "\n".join(lines) if lines else "World context unavailable."


# ── Dispatcher ─────────────────────────────────────────────────────────────────


class ActionDispatcher:
    """Routes a typed slot schema to the correct ``ActionHandler``.

    Maintains a class-level registry of action type string → handler class.
    Instantiating it creates one handler instance per registered type.
    """

    _registry: Dict[str, Type[ActionHandler]] = {}

    @classmethod
    def register(cls, action_type: str, handler_class: Type[ActionHandler]) -> None:
        """Register a handler class for an action type string."""
        cls._registry[action_type] = handler_class
        logger.debug("ActionDispatcher: registered handler for '%s'.", action_type)

    def __init__(self, world: World, world_context: WorldContext) -> None:
        self._handlers: Dict[str, ActionHandler] = {
            action_type: handler_cls(world, world_context)
            for action_type, handler_cls in self._registry.items()
        }

    def dispatch(self, schema: ActionSlotSchema) -> Any:
        """Execute the action described by *schema* and return the result."""
        action_type = schema.action_type
        handler = self._handlers.get(action_type)
        if handler is None:
            raise KeyError(
                f"No handler registered for action type '{action_type}'. "
                f"Registered types: {list(self._handlers)}. "
                "Implement an ActionHandler subclass and call ActionDispatcher.register()."
            )
        logger.info("Dispatching to %s handler.", action_type)
        return handler.execute(schema)


# ── Default registrations ──────────────────────────────────────────────────────

ActionDispatcher.register("PickUpAction", PickUpActionHandler)
ActionDispatcher.register("PlaceAction", PlaceActionHandler)
