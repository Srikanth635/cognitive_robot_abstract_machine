"""
Stepping a scene under MuJoCo physics at a chosen speed, in a window or without one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta

from semantic_digital_twin.adapters.multi_sim import MujocoSim

from experiments.physics_viewer.exceptions import HeadlessRunWithoutDuration
from experiments.physics_viewer.scenes import PhysicsScene

# %% what a run did


@dataclass(frozen=True)
class PhysicsRun:
    """
    What a run of the physics viewer did.
    """

    frames: int
    """
    How many frames the physics was stepped by.
    """

    simulated_seconds: float
    """
    How much simulated time the frames add up to.
    """


# %% running a scene


@dataclass
class PhysicsViewer:
    """
    Puts a scene under MuJoCo physics and steps it frame by frame, as fast as real time
    times the speed, until its duration is over or its window is closed.

    A scene free-running in MuJoCo is stepped as fast as the machine can, which shows a
    dropped cube at rest at once; stepping frame by frame at the pace of the clock shows
    what happens.
    """

    scene: PhysicsScene
    """
    The scene to put under physics.
    """

    headless: bool = False
    """
    Whether to run without a window.
    """

    speed: float = 1.0
    """
    How many times faster than real time the simulated time passes.
    """

    duration: timedelta | None = None
    """
    How much simulated time to run for; without one a run with a window lasts until the
    window is closed.
    """

    frame: timedelta = timedelta(seconds=1 / 60)
    """
    How much simulated time one frame shows.
    """

    def run(self) -> PhysicsRun:
        """
        Step the scene's physics, frame by frame, until the duration is over or the
        window is closed.

        :raises HeadlessRunWithoutDuration: When there is no window and no duration.
        """
        if self.headless and self.duration is None:
            raise HeadlessRunWithoutDuration()
        simulation = MujocoSim(world=self.scene.value.build(), headless=self.headless)
        simulation.start_stepped_simulation()
        frames = 0
        started = time.monotonic()
        try:
            while not self.is_over(simulation, frames):
                simulation.step_simulation(self.frame)
                frames += 1
                time.sleep(
                    max(0.0, started + self.real_seconds_of(frames) - time.monotonic())
                )
        finally:
            simulation.stop_simulation()
        return PhysicsRun(
            frames=frames, simulated_seconds=frames * self.frame.total_seconds()
        )

    def is_over(self, simulation: MujocoSim, frames: int) -> bool:
        """
        :param simulation: The simulation being stepped.
        :param frames: How many frames have been stepped.
        :return: Whether the duration is over, or the window has been closed.
        """
        if self.duration is not None and frames >= round(
            self.duration.total_seconds() / self.frame.total_seconds()
        ):
            return True
        return not self.headless and not simulation.simulator.renderer.is_running()

    def real_seconds_of(self, frames: int) -> float:
        """
        :param frames: How many frames have been stepped.
        :return: How many seconds after the start of the run those frames are due.
        """
        return frames * self.frame.total_seconds() / self.speed
