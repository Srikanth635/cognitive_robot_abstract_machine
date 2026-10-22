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
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)

from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.exceptions import HeadlessRunWithoutDuration
from experiments.physics_viewer.overlay import ActionOverlay
from experiments.physics_viewer.paced_simulation import RealTimeMujocoSim
from experiments.physics_viewer.placement import Position
from experiments.physics_viewer.scenes import PhysicsScene

logger = logging.getLogger(__name__)

# %% what a run did


@dataclass(frozen=True)
class MovedObject:
    """
    An object an action moved.
    """

    body: str
    """
    The name of the object's body.
    """

    before: Position
    """
    Where the object was before the action.
    """

    after: Position
    """
    Where the object was after the action.
    """

    @property
    def distance(self) -> float:
        """
        :return: How far the object moved, in meters.
        """
        return (
            (self.before.x - self.after.x) ** 2
            + (self.before.y - self.after.y) ** 2
            + (self.before.z - self.after.z) ** 2
        ) ** 0.5


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

    moved_objects: tuple[MovedObject, ...] = ()
    """
    The objects that moved during the action; none without an action.
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
            world=world,
            speed=self.speed,
            headless=self.headless,
            **self.scene.value.simulation_options().keywords(),
        )
        simulation.start_stepped_simulation()
        overlay = ActionOverlay.of(self.headless, simulation)
        camera = self.scene.value.initial_camera()
        if camera is not None and not self.headless:
            camera.apply(simulation.simulator.renderer.cam)
        moved: tuple[tuple[MovedJoint, ...], tuple[MovedObject, ...]] = ((), ())
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
            moved_joints=moved[0],
            moved_objects=moved[1],
        )

    def perform(
        self,
        simulation: RealTimeMujocoSim,
        world: World,
        overlay: ActionOverlay | None,
    ) -> tuple[tuple[MovedJoint, ...], tuple[MovedObject, ...]]:
        """
        Let the scene settle, then have the robot perform the action under physics, and say
        which joints it moved.

        :param simulation: The simulation, started.
        :param world: The world being simulated.
        :param overlay: What writes into the window, if there is one.
        :return: The joints and the objects that moved.
        """
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        simulation.step_simulation(self.settling)
        plan = self.action.value.build(
            Context(world=world, robot=robot, evaluate_conditions=False)
        )
        if overlay is not None:
            overlay.show_started(self.action)
        before = self.joint_positions(world)
        objects_before = self.object_positions(world)
        with PhysicallySimulatedRobot(simulation=simulation):
            plan.perform()
        moved = self.joints_moved_since(before, world)
        moved_objects = self.objects_moved_since(objects_before, world)
        logger.info("%s: moved %d joints", self.action.name.lower(), len(moved))
        for joint in moved:
            logger.info("  %s: %.3f -> %.3f", joint.joint, joint.before, joint.after)
        for item in moved_objects:
            logger.info("  %s: moved %.2f m", item.body, item.distance)
        if overlay is not None:
            overlay.show_done(self.action, moved, moved_objects)
        return moved, moved_objects

    @staticmethod
    def object_positions(world: World) -> dict[str, Position]:
        """
        :return: Where every object that moves freely is, by the name of its body.
        """
        positions = {}
        for connection in world.connections:
            if isinstance(connection, Connection6DoF):
                pose = connection.child.global_pose
                positions[connection.child.name.name] = Position(
                    float(pose.x), float(pose.y), float(pose.z)
                )
        return positions

    def objects_moved_since(
        self, before: dict[str, Position], world: World
    ) -> tuple[MovedObject, ...]:
        """
        :param before: Where the objects were, by the names of their bodies.
        :param world: The world they are in.
        :return: The objects that have moved at least :attr:`reported_movement` since.
        """
        after = self.object_positions(world)
        return tuple(
            moved
            for name, position in after.items()
            if (moved := MovedObject(name, before[name], position)).distance
            >= self.reported_movement
        )

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
