"""
``python -m experiments.physics_viewer SCENE``: put a scene under MuJoCo physics in a
window.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

from typing_extensions import Sequence

from experiments.physics_viewer.scenes import PhysicsScene
from experiments.physics_viewer.viewer import PhysicsViewer


def parse_arguments(arguments: Sequence[str]) -> PhysicsViewer:
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
        "--headless",
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
        help="simulated seconds to run for; with a window, by default until it is closed",
    )
    parsed = parser.parse_args(list(arguments))
    return PhysicsViewer(
        scene=PhysicsScene.from_command_line(parsed.scene),
        headless=parsed.headless,
        speed=parsed.speed,
        duration=(
            None if parsed.duration is None else timedelta(seconds=parsed.duration)
        ),
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


if __name__ == "__main__":
    main()
