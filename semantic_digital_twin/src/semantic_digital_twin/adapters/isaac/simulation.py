"""
A world simulated under Isaac Sim's PhysX, stepped in lockstep with the controller that
drives it.
"""

from __future__ import annotations

import math
import tempfile
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from physics_simulators.base_simulator import BaseSimulator, SimulatorState
from physics_simulators.isaac_simulator import IsaacSimulator
from typing_extensions import Any, ClassVar, Type

from semantic_digital_twin.adapters.isaac.exceptions import DurationNotWholeStepsError
from semantic_digital_twin.adapters.isaac.synchronizer import IsaacSynchronizer
from semantic_digital_twin.adapters.multi_sim import (
    MultiSim,
    MultiSimBuilder,
    MultiSimSynchronizer,
    SteppedSimulation,
)
from semantic_digital_twin.adapters.usd.builder import UsdBuilder
from semantic_digital_twin.exceptions import (
    SimulationAlreadyRunningError,
    SimulationNotStartedError,
)
from semantic_digital_twin.world import World

# %% where the stage is written


class TemporaryStage(StrEnum):
    """
    Where a world's stage is written when no file is given for it.
    """

    DIRECTORY_PREFIX = "isaacsim_"
    """
    The start of the name of the temporary directory the stage is written into.
    """

    FILE_NAME = "scene.usdc"
    """
    The name of the stage's file, in USD's binary format.
    """


# %% the simulation


class IsaacSim(MultiSim, SteppedSimulation):
    """
    A world written as a USD stage and simulated under Isaac Sim's PhysX.
    """

    simulator_class: ClassVar[Type[BaseSimulator]] = IsaacSimulator
    synchronizer_class: ClassVar[Type[MultiSimSynchronizer]] = IsaacSynchronizer
    builder_class: ClassVar[Type[MultiSimBuilder]] = UsdBuilder
    simulator: IsaacSimulator
    synchronizer: IsaacSynchronizer

    def __init__(
        self,
        world: World,
        headless: bool = True,
        step_size: float = 1.0 / 200.0,
        file_path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        :param world: The world to simulate.
        :param headless: Whether the simulation runs without a window.
        :param step_size: How much simulated time, in seconds, one physics step advances;
            it divides every duration the simulation is stepped by.
        :param file_path: The file the stage is written to; if not given, a temporary
            one, removed with the simulation.
        :param kwargs: Settings of :class:`IsaacSimulator`, such as its ``device``.
        """
        self._stage_directory = None
        if file_path is None:
            self._stage_directory = tempfile.TemporaryDirectory(
                prefix=TemporaryStage.DIRECTORY_PREFIX
            )
            file_path = Path(self._stage_directory.name) / TemporaryStage.FILE_NAME
        super().__init__(
            world=world,
            headless=headless,
            step_size=step_size,
            file_path=file_path,
            **kwargs,
        )

    def start_stepped_simulation(self) -> None:
        if self.simulator.state == SimulatorState.RUNNING:
            raise SimulationAlreadyRunningError(self.world.root.name.name)
        self.simulator.start(simulate_in_thread=False, render_in_thread=False)
        self.synchronizer.connect()
        self.synchronizer.command_actuators_from_world_state()

    def step_simulation(self, duration: timedelta) -> None:
        """
        :meth:`SteppedSimulation.step_simulation`, advancing only by whole physics
        steps; the world reads the simulation back once they are done.

        :raises DurationNotWholeStepsError: If the duration is not a whole number of
            physics steps.
        """
        if self.simulator.state != SimulatorState.RUNNING:
            raise SimulationNotStartedError(self.world.root.name.name)
        steps = duration.total_seconds() / self.simulator.step_size
        if not math.isclose(steps, round(steps)):
            raise DurationNotWholeStepsError(
                duration.total_seconds(), self.simulator.step_size
            )
        for _ in range(round(steps)):
            self.simulator.step()
        self.simulator.render()
        self.synchronizer.read_back()

    def servo_positions(self, servo_names: list[str]) -> dict[str, float]:
        return self.synchronizer.servo_positions(servo_names)
