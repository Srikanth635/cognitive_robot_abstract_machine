"""
main_bridge.py – Demonstrates SimulationBridge with a mock PyCRAM world.

This script runs without a real robot or simulator.  It uses lightweight
mock objects that satisfy the World/Body/Robot interface expected by
SimulationBridge, so you can verify the full parse → resolve → execute
pipeline offline.

Run:
    cd /home/malineni/workingdir/cognitive_robot_abstract_machine
    python -m llmr.main_bridge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import MagicMock

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("main_bridge")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Minimal mock world that SimulationBridge can talk to
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MockPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __repr__(self) -> str:
        return f"Pose(x={self.x}, y={self.y}, z={self.z})"


@dataclass
class MockBody:
    """Minimal stand-in for semantic_digital_twin Body."""
    name: str
    global_pose: MockPose = field(default_factory=MockPose)

    def __repr__(self) -> str:
        return f"Body(name={self.name!r}, pose={self.global_pose})"


class MockWorld:
    """Minimal stand-in for semantic_digital_twin.world.World."""

    def __init__(self, bodies: List[MockBody]) -> None:
        self.bodies = bodies

    def __repr__(self) -> str:
        names = [b.name for b in self.bodies]
        return f"MockWorld(bodies={names})"


class MockRobot:
    """Minimal stand-in for AbstractRobot."""
    name: str = "pr2"

    def __repr__(self) -> str:
        return "MockRobot(pr2)"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Patch SequentialPlan so we can execute without real PyCRAM
# ─────────────────────────────────────────────────────────────────────────────

class _MockSequentialPlan:
    """Simulates SequentialPlan.perform() by resolving each PartialDesignator."""

    def __init__(self, context: Any, *children: Any) -> None:
        self._context = context
        self._children = children

    def perform(self) -> Any:
        results = []
        for child in self._children:
            if hasattr(child, "resolve"):
                # PartialDesignator.resolve() → ActionDescription instance
                action = child.resolve()
                logger.info("  ↳ resolved to %s", action.__class__.__name__)
                results.append(action)
            else:
                results.append(child)
        return results or None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main demo
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # -- Build a small mock world with the objects the CRAM plan mentions ----
    world = MockWorld(
        bodies=[
            MockBody("cup",   MockPose(x=0.5, y=0.1, z=0.8)),
            MockBody("table", MockPose(x=0.5, y=0.0, z=0.75)),
            MockBody("pr2",   MockPose(x=0.0, y=0.0, z=0.0)),
        ]
    )
    robot = MockRobot()

    print("\n" + "=" * 60)
    print("  SimulationBridge Demo")
    print("=" * 60)
    print(f"\nWorld: {world}")
    print(f"Robot: {robot}\n")

    # -- Create the bridge ---------------------------------------------------
    from llmr.serializers.simulation_bridge import SimulationBridge
    bridge = SimulationBridge(world, robot)

    # -- Inspect world state before executing --------------------------------
    print("Bodies visible to bridge:", bridge.list_bodies())
    print("Snapshot:")
    for name, pose in bridge.snapshot().items():
        print(f"  {name:10s} → {pose}")

    # -- CRAM plan string (as the LLM would generate it) --------------------
    cram_string = (
        "(an action (type PickingUp) "
        "(object (:tag cup (an object (type Artifact size medium color white)))) "
        "(source (a location (on (:tag table (an object (type Surface)))))))"
    )

    print("\n" + "-" * 60)
    print("CRAM input:")
    print(" ", cram_string)

    # -- Step A: parse only (no PyCRAM needed) -------------------------------
    plan = bridge.parse(cram_string)
    print("\nParsed CRAMActionPlan:")
    print(f"  action_type : {plan.action_type}")
    print(f"  object.tag  : {plan.object.tag}")
    print(f"  object.type : {plan.object.semantic_type}")
    print(f"  source.tag  : {plan.source.tag}")

    # -- Step B: resolve bodies from the live mock world --------------------
    resolver = bridge.make_resolver()
    resolved_cup   = resolver(plan.object)
    resolved_table = resolver(plan.source)
    print("\nBody resolution:")
    print(f"  object (cup)   → {resolved_cup}")
    print(f"  source (table) → {resolved_table}")

    # -- Step C: build a PartialDesignator ----------------------------------
    #   (PyCRAM not installed, so we use the serializer directly and
    #    just display what would be passed to PickUpAction.description())
    print("\nBuilding PartialDesignator …")
    try:
        partial = bridge.to_partial_designator(cram_string)
        print(f"  PartialDesignator: {partial}")
    except Exception as exc:
        # Expected when PyCRAM actions are not importable in this environment
        print(f"  (PartialDesignator skipped — PyCRAM not installed: {exc})")

    # -- Step D: simulate execution with the mock plan ----------------------
    print("\nSimulating execute() with MockSequentialPlan …")
    import sys
    # Temporarily inject the mock so bridge.execute() doesn't fail
    import unittest.mock as _mock
    with _mock.patch("llmr.serializers.simulation_bridge.SequentialPlan",
                     _MockSequentialPlan, create=True):
        try:
            bridge.execute(cram_string)
            print("  execute() completed successfully (mock mode)")
        except Exception as exc:
            print(f"  (execute skipped — PyCRAM not installed: {exc})")

    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
