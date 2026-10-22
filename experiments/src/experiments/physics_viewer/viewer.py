"""
Stepping a scene under MuJoCo physics at a chosen speed, in a window or without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.exceptions import HeadlessRunWithoutDuration
from experiments.physics_viewer.paced_simulation import RealTimeMujocoSim
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

    world: World
    """
    The world as the physics left it.
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

    action: PhysicsAction | None = None
    """
    The action the robot performs first; the duration then counts from the end of the
    action.
    """

    settling: timedelta = timedelta(seconds=1)
    """
    How long a scene is left to come to rest before an action starts.
    """

    def run(self) -> PhysicsRun:
        """
        Perform the action, if there is one, then step the scene's physics, frame by
        frame, until the duration is over or the window is closed.

        :raises HeadlessRunWithoutDuration: When there is no window and no duration.
        """
        if self.headless and self.duration is None:
            raise HeadlessRunWithoutDuration()
        world = self.scene.value.build()
        simulation = RealTimeMujocoSim(
            world=world, speed=self.speed, headless=self.headless
        )
        simulation.start_stepped_simulation()
        try:
            if self.action is not None:
                self.perform(simulation, world)
            frames = self.step_until_over(simulation)
        finally:
            simulation.stop_simulation()
        return PhysicsRun(
            frames=frames,
            simulated_seconds=frames * self.frame.total_seconds(),
            world=world,
        )

    def perform(self, simulation: RealTimeMujocoSim, world: World) -> None:
        """
        Let the scene settle, then have the robot perform the action under physics.

        :param simulation: The simulation, started.
        :param world: The world being simulated.
        """
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        simulation.step_simulation(self.settling)
        plan = self.action.value.build(
            Context(world=world, robot=robot, evaluate_conditions=False)
        )
        with PhysicallySimulatedRobot(simulation=simulation):
            plan.perform()

    def step_until_over(self, simulation: RealTimeMujocoSim) -> int:
        """
        Step the physics frame by frame until the duration is over or the window is
        closed.

        :param simulation: The simulation, started.
        :return: How many frames were stepped.
        """
        frames = 0
        while not self.is_over(simulation, frames):
            simulation.step_simulation(self.frame)
            frames += 1
        return frames

    def is_over(self, simulation: RealTimeMujocoSim, frames: int) -> bool:
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
