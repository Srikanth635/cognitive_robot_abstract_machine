"""
``python -m experiments.physics_viewer.isaac SCENE``: put a scene under Isaac Sim's PhysX,
in a window or headless, and record it from several cameras if asked.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from physics_simulators.isaac_application import IsaacApplication
from typing_extensions import Sequence

# %% Isaac Sim's application, which runs before any of its modules is imported


class CommandLineFlag(StrEnum):
    """
    The command line's flags that decide how Isaac Sim's application starts.
    """

    HEADLESS = "--headless"
    """
    Run without a window.
    """


IsaacApplication(headless=CommandLineFlag.HEADLESS in sys.argv).start()

from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.isaac.cameras import SceneCamera
from experiments.physics_viewer.isaac.viewer import IsaacPhysicsViewer
from experiments.physics_viewer.scenes import PhysicsScene

# %% the command line


def parse_arguments(arguments: Sequence[str]) -> IsaacPhysicsViewer:
    """
    :param arguments: The command line after the program.
    :return: The run the command line asks for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        choices=[scene.name.lower() for scene in PhysicsScene],
        help="the scene to put under physics",
    )
    parser.add_argument(
        "--action",
        choices=[action.name.lower() for action in PhysicsAction],
        default=None,
        help="an action the robot performs first, under physics",
    )
    parser.add_argument(
        CommandLineFlag.HEADLESS,
        action="store_true",
        help="run without a window (needs --duration)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="times faster than real time"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="simulated seconds to run for after the action; with a window, by default "
        "until it is closed",
    )
    parser.add_argument(
        "--camera",
        choices=[camera.name.lower() for camera in SceneCamera],
        default=SceneCamera.BEHIND_RIGHT.name.lower(),
        help="the camera the window shows first",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="a directory to write a video of every camera into, with player.html, a "
        "page that plays them",
    )
    parsed = parser.parse_args(list(arguments))
    return IsaacPhysicsViewer(
        scene=PhysicsScene.from_command_line(parsed.scene),
        headless=parsed.headless,
        speed=parsed.speed,
        action=(
            None
            if parsed.action is None
            else PhysicsAction.from_command_line(parsed.action)
        ),
        duration=(
            None if parsed.duration is None else timedelta(seconds=parsed.duration)
        ),
        camera=SceneCamera[parsed.camera.upper()],
        record_directory=parsed.record,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    """
    Run the scene the command line names.

    :param arguments: The command line after the program, or ``None`` to read it as
        given.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    run = parse_arguments(sys.argv[1:] if arguments is None else arguments).run()
    logging.getLogger(__name__).info(
        "stepped %d frames, %.1f simulated seconds", run.frames, run.simulated_seconds
    )
    IsaacApplication().start().close()


if __name__ == "__main__":
    main()
