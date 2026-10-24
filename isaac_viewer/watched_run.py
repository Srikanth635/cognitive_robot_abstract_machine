"""
A run in which the robot of a physics viewer scene performs an action under Isaac Sim
while cameras watch it.

Import it only once Isaac Sim's application runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta

import omni.usd
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import PhysicallySimulatedRobot
from experiments.physics_viewer.actions import PhysicsAction
from experiments.physics_viewer.scenes import PhysicsScene
from pxr import Sdf
from scene_cameras import SceneCamera, SceneLighting

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

# %% the run


@dataclass
class WatchedRun(ABC):
    """
    Builds the scene, lights it, puts up every :class:`SceneCamera`, lets the scene
    settle, and has its robot perform the action; what is done with the cameras is left
    to each kind of run.
    """

    scene: PhysicsScene
    """
    The scene the robot acts in.
    """

    action: PhysicsAction
    """
    What the robot does.
    """

    settling: timedelta = field(default=timedelta(seconds=1))
    """
    The simulated time the scene settles for before the robot acts, as the physics
    viewer gives it.
    """

    lighting: SceneLighting = field(default_factory=SceneLighting)
    """
    The lights the scene is watched under.
    """

    def run(self) -> None:
        """
        Watch the robot perform the action.
        """
        world = self.scene.value.build()
        [robot] = world.get_semantic_annotations_by_type(AbstractRobot)
        simulation = self._simulation_of(world)
        simulation.start_stepped_simulation()
        stage = omni.usd.get_context().get_stage()
        self.lighting.add_to(stage)
        self._watch_with({camera: camera.mount(stage, robot) for camera in SceneCamera})
        simulation.step_simulation(self.settling)
        self._before_the_action()
        plan = self.action.value.build(
            Context(world=world, robot=robot, evaluate_conditions=False)
        )
        with PhysicallySimulatedRobot(simulation=simulation):
            plan.perform()
        self._after_the_action(simulation)
        simulation.stop_simulation()

    @abstractmethod
    def _simulation_of(self, world: World) -> IsaacSim:
        """
        :param world: The scene's world.
        :return: The simulation the world runs in, not started yet.
        """

    @abstractmethod
    def _watch_with(self, cameras: dict[SceneCamera, Sdf.Path]) -> None:
        """
        Start watching, once the cameras are up.

        :param cameras: The prim each camera is written as.
        """

    @abstractmethod
    def _before_the_action(self) -> None:
        """
        Do what the run does once the scene has settled, before the robot acts.
        """

    @abstractmethod
    def _after_the_action(self, simulation: IsaacSim) -> None:
        """
        Do what the run does once the robot has acted, before the simulation stops.

        :param simulation: The simulation, still running.
        """
