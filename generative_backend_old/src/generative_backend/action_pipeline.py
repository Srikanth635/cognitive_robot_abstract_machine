"""Universal action pipeline: any NL instruction → executable pycram action.

This is the single entry point for the generative backend.

## Pipeline flow

    NL instruction
        │
        ▼  slot_filler agent  (classify + extract in one LLM call)
    PickUpSlotSchema | PlaceSlotSchema
        │
        ▼  ActionDispatcher.dispatch()
          │
          ├─ PickUpAction → PickUpActionHandler
          │     Ground object
          │     Build PartialDesignator[PickUpAction]
          │     Resolve arm + grasp via LLM
          │     → PickUpAction
          │
          └─ PlaceAction → PlaceActionHandler
                Ground object + ground target surface
                Build PartialDesignator[PlaceAction]
                Resolve arm via LLM
                → PlaceAction

## Usage

    from generative_backend.action_pipeline import ActionPipeline

    pipeline = ActionPipeline.from_world(world)
    action = pipeline.run("Pick up the red cup from the table")
    action = pipeline.run("Place the mug on the kitchen counter")
    action.perform()

## Adding a new action type

1. Add ``YourActionSlotSchema`` to ``workflows/schemas/``.
2. Add a prompt to ``workflows/prompts/``.
3. Update ``workflows/agents/slot_filler.py`` to classify and return the new schema.
4. Implement ``YourActionHandler`` in ``action_dispatcher.py`` and register it.

The ``ActionPipeline`` requires zero changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.world import World

from pycram.datastructures.enums import Arms
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.placing import PlaceAction

from .action_dispatcher import ActionDispatcher, WorldContext
from .workflows.agents.slot_filler import ActionSlotSchema, run_slot_filler
from .workflows.schemas.pick_up import PickUpSlotSchema
from .workflows.schemas.place import PlaceSlotSchema

logger = logging.getLogger(__name__)


# ── World serialiser ───────────────────────────────────────────────────────────


def _serialise_world_for_llm(world: World) -> str:
    """Produce a concise string description of the world state for LLM context."""
    lines = ["## World State Summary\n"]

    try:
        body_names = [
            str(getattr(getattr(b, "name", None), "name", b))
            for b in world.bodies
        ]
        lines.append(f"Bodies present: {', '.join(body_names[:30])}")
        if len(body_names) > 30:
            lines.append(f"  … and {len(body_names) - 30} more.")
    except Exception:  # noqa: BLE001
        lines.append("Bodies: unavailable")

    try:
        ann_summary: dict[str, list[str]] = {}
        for b in list(world.bodies)[:20]:
            anns = world.get_semantic_annotations_of_body(b)
            if anns:
                b_name = str(getattr(getattr(b, "name", None), "name", b))
                ann_summary[b_name] = [type(a).__name__ for a in anns]
        if ann_summary:
            lines.append("\nSemantic annotations:")
            for body_name, types in ann_summary.items():
                lines.append(f"  {body_name}: {', '.join(types)}")
    except Exception:  # noqa: BLE001
        pass

    return "\n".join(lines)


# ── Pipeline ───────────────────────────────────────────────────────────────────


@dataclass
class ActionPipeline:
    """Universal NL → pycram action pipeline.

    Handles any instruction whose action type is registered in ``ActionDispatcher``.
    No subclassing or per-action configuration required.

    :param world: The Semantic Digital Twin world instance.
    :param world_context: Runtime context carrying the robot manipulator.
    """

    world: World
    world_context: WorldContext

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_world(
        cls,
        world: World,
        arm: Optional[Arms] = None,
    ) -> "ActionPipeline":
        """Create a pipeline from a world instance.

        Auto-detects the robot's manipulator for use in grasp descriptions
        (required for PickUpAction; benign no-op for PlaceAction).

        :param world: SDT world with a loaded robot and objects.
        :param arm: Which arm's manipulator to use.  Defaults to RIGHT.
        """
        manipulator = cls._find_manipulator(world, arm or Arms.RIGHT)
        if manipulator is None:
            logger.warning(
                "Could not auto-detect robot manipulator.  "
                "GraspDescription construction for PickUpAction will fail unless "
                "a manipulator is injected manually into world_context."
            )
        return cls(
            world=world,
            world_context=WorldContext(manipulator=manipulator),
        )

    @staticmethod
    def _find_manipulator(world: World, arm: Arms) -> Optional[Manipulator]:
        """Retrieve the ``Manipulator`` annotation for *arm* from the world."""
        try:
            robots = world.get_semantic_annotations_by_type(AbstractRobot)
            if not robots:
                return None
            robot = robots[0]
            if hasattr(robot, "get_manipulator_for_arm"):
                return robot.get_manipulator_for_arm(arm)
            if hasattr(robot, "manipulators"):
                manipulators = robot.manipulators
                if isinstance(manipulators, dict):
                    return manipulators.get(arm)
                if isinstance(manipulators, list) and manipulators:
                    return manipulators[arm.value] if arm.value < len(manipulators) else manipulators[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not retrieve manipulator: %s", exc)
        return None

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self, instruction: str) -> Union[PickUpAction, PlaceAction]:
        """Execute the full pipeline for *any* supported action type.

        :param instruction: Natural language instruction.
        :return: Fully specified, executable pycram action.
        :raises RuntimeError: On unrecoverable failures in any stage.
        """
        logger.info("ActionPipeline.run: '%s'", instruction)
        schema = self.classify_and_extract(instruction)
        if schema is None:
            raise RuntimeError(
                "Slot-filler failed.  Check LLM connectivity and API keys."
            )
        return self.dispatch(schema)

    # ── Step-by-step accessors (for debugging / notebooks) ────────────────────

    def classify_and_extract(self, instruction: str) -> Optional[ActionSlotSchema]:
        """Phase 1: NL instruction → typed slot schema (classify + extract).

        :return: ``PickUpSlotSchema`` or ``PlaceSlotSchema``; ``None`` on failure.
        """
        world_ctx_str = _serialise_world_for_llm(self.world)
        schema = run_slot_filler(instruction=instruction, world_context=world_ctx_str)
        if schema is not None:
            logger.info(
                "classify_and_extract – action_type=%s, object='%s', arm=%s",
                schema.action_type,
                schema.object_description.name,
                schema.arm,
            )
        return schema

    def dispatch(
        self, schema: Union[PickUpSlotSchema, PlaceSlotSchema]
    ) -> Union[PickUpAction, PlaceAction]:
        """Phase 2: typed slot schema → concrete pycram action (ground + resolve).

        :param schema: Output of ``classify_and_extract()``.
        :return: Fully specified pycram action.
        """
        dispatcher = ActionDispatcher(world=self.world, world_context=self.world_context)
        action = dispatcher.dispatch(schema)
        logger.info("ActionPipeline.dispatch complete – %s resolved.", schema.action_type)
        return action
