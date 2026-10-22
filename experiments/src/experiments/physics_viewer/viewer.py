"""
Stepping a scene under MuJoCo physics at a chosen speed, in a window or without one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.exceptions import HeadlessRunWithoutDuration
from experiments.physics_viewer.overlay import ActionOverlay
from experiments.physics_viewer.paced_simulation import RealTimeMujocoSim
from experiments.physics_viewer.scenes import PhysicsScene

logger = logging.getLogger(__name__)

# %% what a run did


@dataclass(frozen=True)
class MovedJoint:
    """
    A joint an action moved.
    """

    joint: str
    """
    The name of the joint.
    """

    before: float
    """
    Where the joint was before the action, in radians or meters.
    """

    after: float
    """
    Where the joint was after the action, in radians or meters.
    """


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

    moved_joints: tuple[MovedJoint, ...] = ()
    """
    The joints the action moved; none without an action.
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

    reported_movement: float = 0.01
    """
    How far a joint has to have moved, in radians or meters, for the action to be said to
    have moved it.
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
        overlay = ActionOverlay.of(self.headless, simulation)
        camera = self.scene.value.initial_camera()
        if camera is not None and not self.headless:
            camera.apply(simulation.simulator.renderer.cam)
        moved = ()
        try:
            if self.action is not None:
                moved = self.perform(simulation, world, overlay)
            frames = self.step_until_over(simulation)
        finally:
            simulation.stop_simulation()
        return PhysicsRun(
            frames=frames,
            simulated_seconds=frames * self.frame.total_seconds(),
            world=world,
            moved_joints=moved,
        )

    def perform(
        self,
        simulation: RealTimeMujocoSim,
        world: World,
        overlay: ActionOverlay | None,
    ) -> tuple[MovedJoint, ...]:
        """
        Let the scene settle, then have the robot perform the action under physics, and say
        which joints it moved.

        :param simulation: The simulation, started.
        :param world: The world being simulated.
        :param overlay: What writes into the window, if there is one.
        :return: The joints the action moved.
        """
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        simulation.step_simulation(self.settling)
        plan = self.action.value.build(
            Context(world=world, robot=robot, evaluate_conditions=False)
        )
        if overlay is not None:
            overlay.show_started(self.action)
        before = self.joint_positions(world)
        with PhysicallySimulatedRobot(simulation=simulation):
            plan.perform()
        moved = self.joints_moved_since(before, world)
        logger.info("%s: moved %d joints", self.action.name.lower(), len(moved))
        for joint in moved:
            logger.info("  %s: %.3f -> %.3f", joint.joint, joint.before, joint.after)
        if overlay is not None:
            overlay.show_done(self.action, moved)
        return moved

    @staticmethod
    def joint_positions(world: World) -> dict[str, float]:
        """
        :return: Where every joint a servo drives is, by its name; the passive ones, such as
            a tap handle or the motor of a gripper, move with whatever touches them.
        """
        driven = {dof for actuator in world.actuators for dof in actuator.dofs}
        return {
            connection.name.name: float(connection.position)
            for connection in world.connections
            if isinstance(connection, ActiveConnection1DOF) and connection.dof in driven
        }

    def joints_moved_since(
        self, before: dict[str, float], world: World
    ) -> tuple[MovedJoint, ...]:
        """
        :param before: Where the joints were, by their names.
        :param world: The world they are in.
        :return: The joints that have moved at least :attr:`reported_movement` since.
        """
        after = self.joint_positions(world)
        return tuple(
            MovedJoint(joint, before[joint], position)
            for joint, position in after.items()
            if abs(position - before[joint]) >= self.reported_movement
        )

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
