"""
Hooks that attach the bridge to a running coraplex/giskardpy demo.

.. warning:: Every world access (forward kinematics for poses etc.) happens on
   the *simulation* thread itself, inside the ``Executor.tick`` hook. Reading
   the world from a separate sampler thread corrupts the native solver's heap
   — the HTTP handlers therefore only ever serve the last finished snapshot
   dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from coraplex.plans.executors import PlanExecutor
from cramph.executor import StatechartExecutor
from semantic_digital_twin.adapters.mesh import MeshParser
from semantic_digital_twin.world import World

from cram_viz.live.bridge import BRIDGE

logger = logging.getLogger(__name__)


def install_tick_hook(plan_snapshot_tick_interval: int = 5) -> None:
    """
    Bind the bridge to the executing world and snapshot on every sim tick.

    The plan tree is only re-walked every plan_snapshot_tick_interval ticks.
    """
    original_tick = StatechartExecutor.tick

    def tick(self, *args: Any, **kwargs: Any) -> None:
        """
        Run the real tick, then bind/snapshot the bridge off its result.
        """
        result = original_tick(self, *args, **kwargs)
        if BRIDGE.world is None:
            BRIDGE.world = self.context.world
            BRIDGE._bind()
            logger.info(
                "attached to world (robot=%s, %d joints)",
                type(BRIDGE.robot).__name__ if BRIDGE.robot else "?",
                len(BRIDGE._connections),
            )
        try:
            BRIDGE.apply_moves()  # viewer drags land in the real world here
            BRIDGE.snapshot()
            BRIDGE.observe_chart(self.statechart)
            BRIDGE._ticks += 1
            if BRIDGE._ticks % plan_snapshot_tick_interval == 0:
                BRIDGE.snapshot_plan()
        except Exception:
            # boundary guard: a visualization bug must never take the robot
            # demo down — this is the single intentional broad except
            logger.exception("bridge snapshot failed this tick")
        return result

    StatechartExecutor.tick = tick


def install_plan_hooks() -> None:
    """
    Follow the plan an executor performs: publish its statechart when it starts and once
    it ended, however it ended.

    The hook fires on the thread that runs the plan (the same one that ticks the
    executor), so it may touch the statechart directly.
    """
    original_execute = PlanExecutor.execute

    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        Capture the statechart the moment the plan starts, and publish it as it ends.
        """
        BRIDGE._plan = self.statechart
        BRIDGE.snapshot_plan()
        try:
            return original_execute(self, *args, **kwargs)
        finally:
            BRIDGE.snapshot_plan()

    PlanExecutor.execute = execute


def install_mesh_hook() -> None:
    """
    Remember every mesh an object is built from, so the bridge can serve its geometry to
    the viewer.

    All mesh formats go through ``MeshParser.create_world``; the body name matches the
    mesh file's basename.
    """
    original_create_world = MeshParser.create_world

    def create_world(self, *args: Any, **kwargs: Any) -> World:
        """
        Remember this mesh's file path before parsing it.
        """
        if self.file_path:
            BRIDGE._mesh_files[Path(self.file_path).name.lower()] = self.file_path
        return original_create_world(self, *args, **kwargs)

    MeshParser.create_world = create_world
