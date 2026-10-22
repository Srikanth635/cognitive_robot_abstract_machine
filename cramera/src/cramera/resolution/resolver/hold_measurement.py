"""
Measuring how the gripper holds the source container a pour names: once per source and
arm, a process of its own performs the steps a trial performs before it pours, from the
scene's initial world, and writes how the gripper then holds the source.

The process runs this module, so the held source it writes is of the class
:mod:`cramera.resolution.resolver.prediction` declares, which the resolver reads it back as.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Any, Callable, Dict, Mapping

from cramera.resolution import configure_console_logging
from cramera.resolution.resolver.pouring_slot import PouringSlot
from cramera.resolution.resolver.prediction import HeldSource, Holding, PickUpFailed
from cramera.resolution.resolver.scene_world import SceneWorldBuilder

HOLDING_MODULE = __spec__.name
"""
The module a process measuring how the gripper holds the source runs.
"""

HELD_SOURCE_FILE = "held_source.json"
"""
The file such a process writes what it measured to.
"""


@dataclass
class MeasuredHolding(Holding):
    """
    Measures how the gripper holds the source once per source and arm, each in a process
    of its own performing the steps before the pour from the scene's initial world.
    """

    bundle: Path
    """
    The scene's bundle directory, from which the world is built anew.
    """

    run: Callable[..., Any] = subprocess.run
    """
    Runs a measuring process.
    """

    measured: Dict[tuple[str, str], HeldSource] = field(default_factory=dict)
    """
    What was measured so far, by source and arm.
    """

    def held(self, values: Mapping[str, Any]) -> HeldSource:
        key = (
            json.dumps(values[PouringSlot.SOURCE_CONTAINER]),
            values[PouringSlot.ARM],
        )
        if key not in self.measured:
            self.measured[key] = self.measure(values)
        return self.measured[key]

    def measure(self, values: Mapping[str, Any]) -> HeldSource:
        """
        :return: How the gripper holds the source, read from a measuring process.
        :raises PickUpFailed: When the process could not pick the source up.
        """
        with tempfile.TemporaryDirectory() as directory:
            held_file = Path(directory) / HELD_SOURCE_FILE
            finished = self.run(
                [
                    sys.executable,
                    "-m",
                    HOLDING_MODULE,
                    str(self.bundle),
                    json.dumps(dict(values)),
                    str(held_file),
                ],
                capture_output=True,
                text=True,
            )
            if finished.returncode != 0:
                raise PickUpFailed(reason=finished.stderr.strip().splitlines()[-1])
            return HeldSource.from_json(json.loads(held_file.read_text()))


def main() -> None:
    """
    Measure how the gripper holds the source after the steps before a pour, from a
    scene's initial world, and write it to a file.
    """
    configure_console_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("values")
    parser.add_argument("held_file", type=Path)
    arguments = parser.parse_args()
    held = HeldSource.measure_in(
        SceneWorldBuilder.robot_world(arguments.bundle), json.loads(arguments.values)
    )
    arguments.held_file.write_text(json.dumps(held.to_json()))


if __name__ == "__main__":
    main()
