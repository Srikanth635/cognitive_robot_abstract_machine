"""
``python view_action.py SCENE ACTION``: watch the robot of a physics viewer scene perform
an action under Isaac Sim's PhysX, in a window, in real time.

Run it through ``run.sh`` next to this file, which sets up the Python environment Isaac
Sim and the repository's packages run in together. Every camera of
:class:`scene_cameras.SceneCamera` is put up; the window's camera menu switches between
them.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from physics_simulators.isaac_application import IsaacApplication
from typing_extensions import Any, Sequence

# %% Isaac Sim's application, which runs before any of its modules is imported

IsaacApplication(headless=False).start()

from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.scenes import PhysicsScene
from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pxr import Sdf
from scene_cameras import SceneCamera
from watched_run import WatchedRun

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.world import World

# %% the command line


def parse_arguments(arguments: Sequence[str]) -> IsaacWindowRun:
    """
    :param arguments: The command line after the program.
    :return: The run the command line asks for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene", help="a physics viewer scene, such as pr2_at_the_counter"
    )
    parser.add_argument(
        "action", help="an action the scene's robot performs, such as go_to_the_counter"
    )
    parser.add_argument(
        "--camera",
        choices=[camera.name.lower() for camera in SceneCamera],
        default=SceneCamera.BEHIND_RIGHT.name.lower(),
        help="the camera the window shows first",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=30.0,
        help="seconds to keep the window open after the action; closing it ends sooner",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="a directory to save a screenshot of the window into before and after the action",
    )
    parsed = parser.parse_args(list(arguments))
    return IsaacWindowRun(
        scene=PhysicsScene.from_command_line(parsed.scene),
        action=PhysicsAction.from_command_line(parsed.action),
        camera=SceneCamera[parsed.camera.upper()],
        hold=timedelta(seconds=parsed.hold),
        capture_directory=parsed.capture,
    )


# %% names in Isaac Sim's window


class Screenshot(StrEnum):
    """
    The screenshots of the window a run saves, by the moment they show.
    """

    BEFORE_THE_ACTION = "before.png"
    """
    The scene settled, before the robot acts.
    """

    AFTER_THE_ACTION = "after.png"
    """
    The scene once the robot has acted.
    """


# %% a simulation shown in a window as it runs


class RealTimeIsaacSim(IsaacSim):
    """
    An :class:`IsaacSim` with a window that is redrawn after every step, stepped no faster
    than real time.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        :param kwargs: The settings of :class:`IsaacSim`; the simulation has a window.
        """
        super().__init__(headless=False, **kwargs)
        self._wall_time_at_start = time.time()

    def start_stepped_simulation(self) -> None:
        super().start_stepped_simulation()
        self._wall_time_at_start = time.time()

    def step_simulation(self, duration: timedelta) -> None:
        """
        :meth:`IsaacSim.step_simulation`, then redraw the window without stepping the
        physics again, and wait until as much real time has passed as simulated time.
        """
        super().step_simulation(duration)
        RenderingManager.render()
        elapsed = time.time() - self._wall_time_at_start
        ahead = self.simulator.current_simulation_time - elapsed
        if ahead > 0:
            time.sleep(ahead)


# %% watching an action in the window


@dataclass
class IsaacWindowRun(WatchedRun):
    """
    A run shown in Isaac Sim's window.
    """

    camera: SceneCamera = SceneCamera.BEHIND_RIGHT
    """
    The camera the window shows first.
    """

    hold: timedelta = field(default=timedelta(seconds=30))
    """
    How long the window stays open after the action.
    """

    capture_directory: Path | None = None
    """
    Where screenshots before and after the action are saved; ``None`` saves none.
    """

    frames_to_open_the_window: int = 30
    """
    How many times the window is drawn before the run starts, so it is open and shows
    the scene.
    """

    frames_to_save_a_screenshot: int = 10
    """
    How many times the window is drawn after a screenshot is asked for, so it is saved.
    """

    hold_step: timedelta = field(default=timedelta(milliseconds=50))
    """
    How much simulated time each step advances while the window stays open.
    """

    def _simulation_of(self, world: World) -> IsaacSim:
        return RealTimeIsaacSim(world=world)

    def _watch_with(self, cameras: dict[SceneCamera, Sdf.Path]) -> None:
        ViewportManager.set_camera(str(cameras[self.camera]))
        for _ in range(self.frames_to_open_the_window):
            RenderingManager.render()

    def _before_the_action(self) -> None:
        self._capture(Screenshot.BEFORE_THE_ACTION)

    def _after_the_action(self, simulation: IsaacSim) -> None:
        self._capture(Screenshot.AFTER_THE_ACTION)
        application = IsaacApplication().start()
        end = time.time() + self.hold.total_seconds()
        while time.time() < end and application.is_running():
            simulation.step_simulation(self.hold_step)

    def _capture(self, screenshot: Screenshot) -> None:
        """
        Save what the window shows, if screenshots are asked for.

        :param screenshot: The screenshot to save.
        """
        if self.capture_directory is None:
            return
        capture_viewport_to_file(
            get_active_viewport(), str(self.capture_directory / screenshot)
        )
        for _ in range(self.frames_to_save_a_screenshot):
            RenderingManager.render()


# %% the program


def main(arguments: Sequence[str] | None = None) -> None:
    """
    Show the run the command line asks for.

    :param arguments: The command line after the program, or ``None`` to read it as
        given.
    """
    parse_arguments(sys.argv[1:] if arguments is None else arguments).run()
    IsaacApplication().start().close()


if __name__ == "__main__":
    main()
