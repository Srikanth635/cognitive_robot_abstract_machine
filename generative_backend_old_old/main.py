"""Smoke-test and demo for the generative_backend pipeline.

Run from the generative_backend/ directory (or repo root after installing):

    python generative_backend/main.py

The script is split into three independent test sections so you can run only
the parts that match your environment:

  Section 1 – Phase 1 slot filler only (needs OPENAI_API_KEY, no SDT/pycram)
  Section 2 – Phase 2 discrete resolver only (needs OPENAI_API_KEY, no SDT/pycram)
  Section 3 – Full end-to-end with a mock world (needs pycram + semantic_digital_twin)
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys

from dotenv import load_dotenv

# Load .env from the generative_backend/ directory (where this file lives)
# before anything else so that OPENAI_API_KEY is available immediately.
_ENV_FILE = pathlib.Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("generative_backend.main")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Phase 1 – Slot filling (no SDT/pycram required)
# ─────────────────────────────────────────────────────────────────────────────

def test_slot_filler() -> None:
    """Test Phase 1: NL instruction → PickUpSlotSchema."""
    print("\n" + "=" * 60)
    print("SECTION 1 — Phase 1: Slot Filler")
    print("=" * 60)

    from generative_backend.workflows.agents.slot_filler import run_slot_filler

    test_cases = [
        # (description, instruction)
        (
            "Fully specified",
            "Pick up the red cup from the table with your left arm, approaching from the front.",
        ),
        (
            "Arm only specified",
            "Grab the mug using the right arm.",
        ),
        (
            "Nothing specified except object",
            "Pick up the bottle.",
        ),
        (
            "Spatial context given",
            "Please pick up the milk carton that is on the kitchen counter.",
        ),
        (
            "Grasp direction only",
            "Take the bowl from above.",
        ),
    ]

    for description, instruction in test_cases:
        print(f"\n--- {description} ---")
        print(f"Instruction: \"{instruction}\"")

        schema = run_slot_filler(instruction)
        if schema is None:
            print("  ERROR: slot_filler returned None (check API key / connectivity)")
            continue

        print(f"  object.name         : {schema.object_description.name}")
        print(f"  object.semantic_type: {schema.object_description.semantic_type}")
        print(f"  object.spatial_ctx  : {schema.object_description.spatial_context}")
        print(f"  object.attributes   : {schema.object_description.attributes}")
        print(f"  arm                 : {schema.arm}")
        if schema.grasp_params:
            gp = schema.grasp_params
            print(f"  grasp.approach      : {gp.approach_direction}")
            print(f"  grasp.vertical      : {gp.vertical_alignment}")
            print(f"  grasp.rotate        : {gp.rotate_gripper}")
        else:
            print(f"  grasp_params        : null (underspecified – will be resolved in Phase 2)")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Phase 2 – Discrete resolver (no SDT/pycram required)
# ─────────────────────────────────────────────────────────────────────────────

def test_discrete_resolver() -> None:
    """Test Phase 2 discrete resolver in isolation with a hand-crafted world context."""
    print("\n" + "=" * 60)
    print("SECTION 2 — Phase 2: Discrete Resolver")
    print("=" * 60)

    from generative_backend.workflows.agents.discrete_resolver import run_discrete_resolver

    # Simulate the world context string that WorldContextBuilder would produce
    world_context = """\
Robot position: x=0.000, y=0.000, z=0.000
Object 'red_cup_1': x=0.800, y=-0.300, z=0.750
  → Object is 0.80m in front of and 0.30m to the right of the robot, 0.75m above robot origin.
  → Semantic types: Artifact, GraspableObject
  → Bounding box (w×d×h): 0.080 × 0.080 × 0.120 m
"""

    test_cases = [
        (
            "Only arm is known",
            "arm = LEFT",
            "approach_direction  (FRONT / BACK / LEFT / RIGHT)\n"
            "vertical_alignment  (TOP / BOTTOM / NoAlignment)\n"
            "rotate_gripper      (true / false)",
        ),
        (
            "Nothing is known",
            "None – all discrete parameters are unspecified.",
            "arm  (choose LEFT or RIGHT based on object position)\n"
            "approach_direction  (FRONT / BACK / LEFT / RIGHT)\n"
            "vertical_alignment  (TOP / BOTTOM / NoAlignment)\n"
            "rotate_gripper      (true / false)",
        ),
    ]

    for description, known, missing in test_cases:
        print(f"\n--- {description} ---")
        resolution = run_discrete_resolver(
            world_context=world_context,
            known_parameters=known,
            parameters_to_resolve=missing,
        )
        if resolution is None:
            print("  ERROR: resolver returned None (check API key / connectivity)")
            continue

        print(f"  arm               : {resolution.arm}")
        print(f"  approach_direction: {resolution.approach_direction}")
        print(f"  vertical_alignment: {resolution.vertical_alignment}")
        print(f"  rotate_gripper    : {resolution.rotate_gripper}")
        print(f"  reasoning         : {resolution.reasoning}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Full pipeline with a mock world (requires pycram + SDT)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_with_mock_world() -> None:
    """Test the full pipeline using a minimal mock SDT world.

    Requires:  pycram, semantic_digital_twin installed in the environment.
    """
    print("\n" + "=" * 60)
    print("SECTION 3 — Full Pipeline (mock SDT world)")
    print("=" * 60)

    # ── Build a minimal mock world ────────────────────────────────────────────
    # We create a bare-minimum mock so the test does not depend on a running
    # simulation.  Replace this with a real World instance in production.

    try:
        from unittest.mock import MagicMock, patch
        from semantic_digital_twin.world import World
        from semantic_digital_twin.world_description.world_entity import Body
    except ImportError as exc:
        print(f"  SKIP: SDT not installed ({exc})")
        return

    try:
        from pycram.robot_plans.actions.core.pick_up import PickUpAction
        from pycram.datastructures.enums import Arms
    except ImportError as exc:
        print(f"  SKIP: pycram not installed ({exc})")
        return

    # -- Mock Body (represents "red_cup")
    cup = MagicMock(spec=Body)
    cup.name = MagicMock()
    cup.name.name = "red_cup_1"
    cup.index = 42

    cup_pose_translation = MagicMock()
    cup_pose_translation.x = 0.8
    cup_pose_translation.y = -0.3
    cup_pose_translation.z = 0.75
    cup_pose = MagicMock()
    cup_pose.translation = cup_pose_translation
    cup.global_pose = cup_pose

    # bounding box mock
    bb = MagicMock()
    bb.dimensions = [0.08, 0.08, 0.12]
    bb_col = MagicMock()
    bb_col.bounding_box.return_value = bb
    cup.collision.as_bounding_box_collection_in_frame.return_value = bb_col

    # -- Mock World
    world = MagicMock(spec=World)
    world.bodies = [cup]
    world.get_semantic_annotations_of_body.return_value = []
    world.get_semantic_annotations_by_type.return_value = []

    # -- Mock Manipulator
    from semantic_digital_twin.robots.abstract_robot import Manipulator
    manipulator = MagicMock(spec=Manipulator)
    manipulator._world = world

    # ── Run the pipeline ──────────────────────────────────────────────────────
    from generative_backend.pipeline import PickUpPipeline
    from generative_backend.partial_designator_builder import WorldContext

    pipeline = PickUpPipeline(
        world=world,
        world_context=WorldContext(manipulator=manipulator),
        eql_preferred=False,  # use fuzzy grounding since krrood may not be installed
    )

    # Test step-by-step accessors first
    instruction = "Pick up the red cup from the table."
    print(f"\nInstruction: \"{instruction}\"")

    # Phase 1a: slot filling
    print("\n[Phase 1a] Slot filling...")
    slot_schema = pipeline.extract_slots(instruction)
    if slot_schema is None:
        print("  ERROR: slot_filler returned None")
        return
    print(f"  Extracted object: '{slot_schema.object_description.name}'")
    print(f"  arm              : {slot_schema.arm}  (null = will be resolved by LLM)")
    print(f"  grasp_params     : {slot_schema.grasp_params}  (null = will be resolved by LLM)")

    # Phase 1b: entity grounding
    print("\n[Phase 1b] Entity grounding...")
    grounding = pipeline.ground(slot_schema)
    print(f"  Found {len(grounding.bodies)} body/bodies (EQL={grounding.used_eql})")
    if grounding.warning:
        print(f"  Warning: {grounding.warning}")
    for b in grounding.bodies:
        print(f"    → {b.name.name}")

    if not grounding.bodies:
        print("  ERROR: no bodies found – check object name in instruction")
        return

    # Build PartialDesignator
    print("\n[Build] PartialDesignator...")
    partial = pipeline.build_partial(slot_schema, grounding)
    missing = partial.missing_parameter()
    print(f"  Missing parameters: {missing if missing else 'none (already fully specified)'}")

    # Phase 2: hybrid resolution
    print("\n[Phase 2] Hybrid resolution (LLM discrete)...")
    try:
        action = pipeline.resolve(partial)
        print(f"  PickUpAction created!")
        print(f"  object_designator : {action.object_designator.name.name}")
        print(f"  arm               : {action.arm}")
        print(f"  approach_direction: {action.grasp_description.approach_direction}")
        print(f"  vertical_alignment: {action.grasp_description.vertical_alignment}")
        print(f"  rotate_gripper    : {action.grasp_description.rotate_gripper}")
    except Exception as exc:
        print(f"  ERROR in Phase 2: {exc}")
        logger.exception("Phase 2 error")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print(
            "\nWARNING: OPENAI_API_KEY not set.\n"
            "LLM calls will fail.  Set the key in generative_backend/.env\n"
            "or export it before running:\n"
            "    export OPENAI_API_KEY=sk-...\n"
        )

    # Run each section.  Comment out sections you don't need.
    try:
        test_slot_filler()
    except Exception as exc:
        logger.error("Section 1 failed: %s", exc, exc_info=True)

    try:
        test_discrete_resolver()
    except Exception as exc:
        logger.error("Section 2 failed: %s", exc, exc_info=True)

    try:
        test_full_pipeline_with_mock_world()
    except Exception as exc:
        logger.error("Section 3 failed: %s", exc, exc_info=True)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
