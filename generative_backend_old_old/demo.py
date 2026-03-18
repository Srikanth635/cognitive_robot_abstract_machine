"""Stage-by-stage transformation trace for a pick-up instruction.

Run from the generative_backend directory:

    python demo.py

Requires OPENAI_API_KEY in generative_backend/.env (or as an env var).

The demo uses a lightweight mock world so no ROS / simulation stack is
needed.  Each pipeline stage is shown with its inputs and outputs.
"""

import json
import logging
import os
import pathlib
import textwrap
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

# ── Load .env before any LLM imports ──────────────────────────────────────────
_ENV_FILE = pathlib.Path(__file__).parent / ".env"
load_dotenv(_ENV_FILE, override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ── Check API key ─────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit(
        "ERROR: OPENAI_API_KEY not set.\n"
        f"Create {_ENV_FILE} with:  OPENAI_API_KEY=sk-..."
    )

# ── Instruction under test ────────────────────────────────────────────────────
INSTRUCTION = "Pick up the cup from the table"


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 70
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def _sub(label: str, value: str) -> None:
    print(f"\n  ▶ {label}")
    for line in value.splitlines():
        print(f"      {line}")


def _json_dump(obj) -> str:
    """Return a compact JSON-like representation of a pydantic model or dict."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    elif hasattr(obj, "__dict__"):
        data = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        return str(obj)
    return json.dumps(data, indent=2, default=str)


# ── Mock world ────────────────────────────────────────────────────────────────

def _make_mock_world():
    """Build a minimal mock of the SDT World with one robot and one cup body."""

    # --- Cup body ---
    cup_name = MagicMock()
    cup_name.name = "cup_1"
    cup_body = MagicMock()
    cup_body.name = cup_name
    cup_pose = MagicMock()
    cup_pose.translation.x = 1.0
    cup_pose.translation.y = -0.1
    cup_pose.translation.z = 0.85
    cup_body.global_pose = cup_pose
    cup_body.collision = MagicMock()  # bounding box calls will safely fail

    # --- Robot body ---
    robot_name = MagicMock()
    robot_name.name = "pr2"
    robot_body = MagicMock()
    robot_body.name = robot_name
    robot_pose = MagicMock()
    robot_pose.translation.x = 0.0
    robot_pose.translation.y = 0.0
    robot_pose.translation.z = 0.0
    robot_body.global_pose = robot_pose

    # --- Manipulator ---
    manipulator = MagicMock()
    manipulator.__class__.__name__ = "Manipulator"

    # --- AbstractRobot annotation ---
    robot_ann = MagicMock()
    robot_ann._robot = MagicMock()
    robot_ann._robot.root = robot_body
    robot_ann.manipulators = {MagicMock(): manipulator}

    # --- World ---
    world = MagicMock()
    world.bodies = [robot_body, cup_body]
    world.get_semantic_annotations_by_type.return_value = [robot_ann]
    world.get_semantic_annotations_of_body.return_value = []
    return world, manipulator


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1a – Slot filling
# ══════════════════════════════════════════════════════════════════════════════

def stage_1a_slot_filling():
    _section("STAGE 1a  ·  Slot Filling   (NL → PickUpSlotSchema)")

    from generative_backend.workflows.prompts.pick_up_prompts import (
        pick_up_slot_filler_prompt,
    )
    from generative_backend.workflows.agents.slot_filler import run_slot_filler

    # Show the rendered prompt
    rendered = pick_up_slot_filler_prompt.format_messages(
        instruction=INSTRUCTION,
    )
    system_msg = rendered[0].content
    human_msg = rendered[1].content

    _sub("System prompt (condensed)", textwrap.shorten(system_msg, width=300, placeholder=" …"))
    _sub("Human message sent to LLM", human_msg)

    print("\n  ⏳ Calling LLM …")
    slot_schema = run_slot_filler(instruction=INSTRUCTION)

    if slot_schema is None:
        print("  ✗  LLM returned None – check API key / connectivity.")
        return None

    _sub("LLM output  →  PickUpSlotSchema", _json_dump(slot_schema))
    return slot_schema


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1b – Entity grounding
# ══════════════════════════════════════════════════════════════════════════════

def stage_1b_entity_grounding(slot_schema, world):
    _section("STAGE 1b  ·  Entity Grounding   (PickUpSlotSchema → Body)")

    from generative_backend.entity_grounder import EntityGrounder

    desc = slot_schema.object_description
    _sub("Entity description from slot schema", _json_dump(desc))

    grounder = EntityGrounder(world, eql_preferred=True)
    result = grounder.ground(desc)

    body_names = [
        str(getattr(getattr(b, "name", None), "name", b)) for b in result.bodies
    ]
    _sub("Grounding result", (
        f"bodies found : {body_names}\n"
        f"used EQL     : {result.used_eql}\n"
        f"warning      : {result.warning or 'none'}"
    ))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1c – Build PartialDesignator
# ══════════════════════════════════════════════════════════════════════════════

def stage_1c_partial_designator(slot_schema, grounding_result, world_context):
    _section("STAGE 1c  ·  PartialDesignator   (SlotSchema + Bodies → PartialDesignator)")

    from generative_backend.partial_designator_builder import build_partial_designator

    partial = build_partial_designator(
        slot_schema=slot_schema,
        grounded_bodies=grounding_result.bodies,
        world_context=world_context,
    )

    # Summarise the partial designator
    filled = {k: str(v) for k, v in partial.kwargs.items() if v is not None}
    missing = partial.missing_parameter()

    _sub("Parameters already filled", json.dumps(filled, indent=2))
    _sub("Missing (free) parameters", str(missing) if missing else "none – fully specified")

    return partial


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 – Hybrid discrete resolution
# ══════════════════════════════════════════════════════════════════════════════

def stage_2_resolution(partial, world, world_context):
    _section("STAGE 2  ·  Hybrid Resolution   (PartialDesignator → PickUpAction)")

    from generative_backend.workflows.prompts.resolution_prompts import (
        pick_up_resolver_prompt,
    )
    from generative_backend.hybrid_resolver import (
        HybridPickUpResolver,
        WorldContextBuilder,
        _known_params_summary,
        _missing_params_description,
    )

    # Build the context strings that go to the LLM
    ctx_builder = WorldContextBuilder(world)
    world_ctx_str = ctx_builder.build_for_pickup(partial)
    known_str = _known_params_summary(partial)
    missing_str = _missing_params_description(partial)

    _sub("World context sent to resolver LLM", world_ctx_str)
    _sub("Known parameters", known_str)
    _sub("Parameters to resolve", missing_str)

    # Show the rendered prompt (human part)
    rendered = pick_up_resolver_prompt.format_messages(
        world_context=world_ctx_str,
        known_parameters=known_str,
        parameters_to_resolve=missing_str,
    )
    _sub("Resolver human message", rendered[1].content)

    print("\n  ⏳ Calling resolver LLM …")

    from generative_backend.workflows.agents.discrete_resolver import run_discrete_resolver
    resolution = run_discrete_resolver(
        world_context=world_ctx_str,
        known_parameters=known_str,
        parameters_to_resolve=missing_str,
    )

    if resolution is None:
        print("  ✗  Resolver LLM returned None – check API key / connectivity.")
        return None

    _sub("LLM resolution  →  PickUpDiscreteResolutionSchema", _json_dump(resolution))

    # Build the final action
    resolver = HybridPickUpResolver(world=world, world_context=world_context)
    action = resolver._build_action(partial, resolution)

    return action


# ══════════════════════════════════════════════════════════════════════════════
# Final output
# ══════════════════════════════════════════════════════════════════════════════

def show_final_action(action):
    _section("FINAL OUTPUT  ·  PickUpAction")

    # Summarise the action fields
    lines = [
        f"object_designator : {getattr(getattr(action.object_designator, 'name', None), 'name', action.object_designator)}",
        f"arm               : {action.arm}",
    ]
    gd = action.grasp_description
    if gd is not None:
        lines += [
            f"approach_direction: {gd.approach_direction}",
            f"vertical_alignment: {gd.vertical_alignment}",
            f"rotate_gripper    : {gd.rotate_gripper}",
            f"manipulator       : {type(gd.manipulator).__name__}",
        ]
    _sub("PickUpAction fields", "\n".join(lines))
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print(f"  Instruction : \"{INSTRUCTION}\"")

    world, manipulator = _make_mock_world()

    # WorldContext carries the manipulator runtime object
    from generative_backend.partial_designator_builder import WorldContext
    world_context = WorldContext(manipulator=manipulator)

    # Patch AbstractRobot import so HybridResolver can query the mock world
    # without needing the full SDT stack.
    from semantic_digital_twin.robots.abstract_robot import AbstractRobot

    # ── Stage 1a ──────────────────────────────────────────────────────────────
    slot_schema = stage_1a_slot_filling()
    if slot_schema is None:
        return

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    grounding_result = stage_1b_entity_grounding(slot_schema, world)
    if not grounding_result.bodies:
        print("  ✗  Entity grounding failed – no bodies found.")
        return

    # ── Stage 1c ──────────────────────────────────────────────────────────────
    partial = stage_1c_partial_designator(slot_schema, grounding_result, world_context)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    action = stage_2_resolution(partial, world, world_context)
    if action is None:
        return

    # ── Final ─────────────────────────────────────────────────────────────────
    show_final_action(action)


if __name__ == "__main__":
    main()
