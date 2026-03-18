"""End-to-end pipeline: NL instruction → PickUpAction.

This module wires Phase 1 and Phase 2 together into a single callable that
takes a natural language instruction and returns an executable ``PickUpAction``.

## Pipeline flow

    NL instruction
        │
        ▼ Phase 1a: slot filling
    PickUpSlotSchema  (LLM extracts what it knows, nulls what it doesn't)
        │
        ▼ Phase 1b: entity grounding
    PartialDesignator[PickUpAction]  (Body resolved via krrood EQL / fuzzy)
        │
        ▼ Phase 2: hybrid resolution
    PickUpAction  (LLM fills discrete params; probabilistic fills continuous)

## Usage

Basic usage with a world that already has a robot loaded::

    from generative_backend.pipeline import PickUpPipeline

    pipeline = PickUpPipeline.from_world(world)
    action = pipeline.run("Pick up the red cup from the table")
    action.perform()

With a known arm (avoids LLM choosing)::

    action = pipeline.run("Pick up the mug using the right arm")

## Extension
To support additional action types (PlaceAction, NavigateAction, …), follow the
same pattern:
  1. Add a slot schema in ``workflows/pydantics/``
  2. Add a prompt in ``workflows/prompts/``
  3. Add a resolution schema and prompt
  4. Add agent graphs in ``workflows/agents/``
  5. Add a specialised pipeline class here
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.world import World

from pycram.datastructures.enums import Arms
from pycram.datastructures.partial_designator import PartialDesignator
from pycram.robot_plans.actions.core.pick_up import PickUpAction

from .entity_grounder import EntityGrounder, GroundingResult
from .hybrid_resolver import HybridPickUpResolver
from .partial_designator_builder import WorldContext, build_partial_designator
from .workflows.agents.slot_filler import run_slot_filler
from .workflows.pydantics.pick_up_schemas import PickUpSlotSchema

logger = logging.getLogger(__name__)


# ── World serialiser (for LLM context strings) ─────────────────────────────────


def _serialise_world_for_llm(world: World) -> str:
    """Produce a concise string description of the world state for LLM context.

    Reuses the belief-state overview pattern from ``llmr.workflows.agents.pycram_mapper``
    but queries the world object directly instead of HTTP endpoints.
    """
    lines = ["## World State Summary\n"]

    # Bodies
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

    # Semantic annotations summary
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


# ── Pipeline class ─────────────────────────────────────────────────────────────


@dataclass
class PickUpPipeline:
    """End-to-end NL → PickUpAction pipeline.

    :param world: The Semantic Digital Twin world instance.
    :param world_context: Runtime context holding the robot manipulator.
    """

    world: World
    world_context: WorldContext

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_world(
        cls,
        world: World,
        arm: Optional[Arms] = None,
    ) -> "PickUpPipeline":
        """Create a pipeline from a world instance.

        Attempts to find the robot's manipulator automatically via SDT semantic
        annotations.  If the robot has multiple arms, ``arm`` selects which
        manipulator to use (defaults to ``Arms.RIGHT``).

        :param world: SDT world with a loaded robot and objects.
        :param arm: Which arm's manipulator to use.  Defaults to RIGHT.
        """
        manipulator = cls._find_manipulator(world, arm or Arms.RIGHT)
        if manipulator is None:
            logger.warning(
                "Could not auto-detect robot manipulator.  "
                "GraspDescription construction will fail unless manipulator is "
                "injected manually into world_context."
            )
        return cls(
            world=world,
            world_context=WorldContext(manipulator=manipulator),
        )

    @staticmethod
    def _find_manipulator(world: World, arm: Arms) -> Optional[Manipulator]:
        """Retrieve the ``Manipulator`` semantic annotation for *arm* from the world."""
        try:
            robots = world.get_semantic_annotations_by_type(AbstractRobot)
            if not robots:
                return None
            robot = robots[0]
            # AbstractRobot provides get_manipulator_for_arm or similar API;
            # the exact method depends on the SDT version – try common patterns.
            if hasattr(robot, "get_manipulator_for_arm"):
                return robot.get_manipulator_for_arm(arm)
            if hasattr(robot, "manipulators"):
                manipulators = robot.manipulators
                # manipulators is typically a dict {Arms: Manipulator} or a list
                if isinstance(manipulators, dict):
                    return manipulators.get(arm)
                if isinstance(manipulators, list) and manipulators:
                    return manipulators[arm.value] if arm.value < len(manipulators) else manipulators[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not retrieve manipulator: %s", exc)
        return None

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self, instruction: str) -> PickUpAction:
        """Execute the full pipeline and return a ``PickUpAction``.

        :param instruction: Natural language pick-up instruction.
        :return: Fully specified, executable ``PickUpAction``.
        :raises RuntimeError: On unrecoverable errors in any stage.
        """
        logger.info("Pipeline starting for instruction: '%s'", instruction)

        # ── Phase 1a: Slot filling ─────────────────────────────────────────────
        world_ctx_str = _serialise_world_for_llm(self.world)
        slot_schema: Optional[PickUpSlotSchema] = run_slot_filler(
            instruction=instruction,
            world_context=world_ctx_str,
        )

        if slot_schema is None:
            raise RuntimeError(
                "Phase 1 slot-filler failed.  Check LLM connectivity and API keys."
            )

        logger.info(
            "Phase 1 complete – object='%s', arm=%s, grasp_params=%s",
            slot_schema.object_description.name,
            slot_schema.arm,
            slot_schema.grasp_params,
        )

        # ── Phase 1b: Entity grounding ────────────────────────────────────────
        grounder = EntityGrounder(self.world)
        grounding_result: GroundingResult = grounder.ground(
            slot_schema.object_description
        )

        if grounding_result.warning:
            logger.warning("Grounding warning: %s", grounding_result.warning)

        if not grounding_result.bodies:
            raise RuntimeError(
                f"Entity grounding failed: no Body found for "
                f"'{slot_schema.object_description.name}'.  "
                f"Check that the object exists in the world."
            )

        logger.info(
            "Phase 1b complete – grounded %d body/bodies (tier=%s).",
            len(grounding_result.bodies),
            grounding_result.tier,
        )

        # ── Build PartialDesignator ────────────────────────────────────────────
        partial: PartialDesignator[PickUpAction] = build_partial_designator(
            slot_schema=slot_schema,
            grounded_bodies=grounding_result.bodies,
            world_context=self.world_context,
        )

        missing = partial.missing_parameter()
        logger.info(
            "PartialDesignator built – missing params: %s",
            missing if missing else "none (fully specified)",
        )

        # ── Phase 2: Hybrid resolution ────────────────────────────────────────
        resolver = HybridPickUpResolver(
            world=self.world,
            world_context=self.world_context,
        )
        action: PickUpAction = resolver.resolve(partial)

        logger.info("Pipeline complete – PickUpAction resolved.")
        return action

    # ── Step-by-step accessors (for debugging / testing) ─────────────────────

    def extract_slots(self, instruction: str) -> Optional[PickUpSlotSchema]:
        """Run Phase 1a only: NL → PickUpSlotSchema."""
        world_ctx_str = _serialise_world_for_llm(self.world)
        return run_slot_filler(instruction=instruction, world_context=world_ctx_str)

    def ground(self, slot_schema: PickUpSlotSchema) -> GroundingResult:
        """Run Phase 1b only: PickUpSlotSchema → GroundingResult."""
        grounder = EntityGrounder(self.world)
        return grounder.ground(slot_schema.object_description)

    def build_partial(
        self,
        slot_schema: PickUpSlotSchema,
        grounding_result: GroundingResult,
    ) -> PartialDesignator[PickUpAction]:
        """Build the PartialDesignator from slot schema + grounding result."""
        return build_partial_designator(
            slot_schema=slot_schema,
            grounded_bodies=grounding_result.bodies,
            world_context=self.world_context,
        )

    def resolve(
        self,
        partial: PartialDesignator[PickUpAction],
    ) -> PickUpAction:
        """Run Phase 2 only: PartialDesignator → PickUpAction."""
        resolver = HybridPickUpResolver(
            world=self.world,
            world_context=self.world_context,
        )
        return resolver.resolve(partial)
