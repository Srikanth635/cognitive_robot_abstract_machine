"""
Stepping the physics of a USD stage in Isaac Sim, and reading and writing its joints and
free bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import omni.replicator.core as replicator
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from omni.replicator.core.scripts.annotators import Annotator
from pxr import UsdPhysics
from typing_extensions import Iterable

from physics_simulators.base_simulator import (
    BaseSimulator,
    SimulatorConstraints,
    SimulatorRenderer,
)
from physics_simulators.exceptions import SteppedOnAnotherThreadError
from physics_simulators.isaac_application import IsaacApplication

# %% where PhysX simulates


class PhysicsDevice(StrEnum):
    """
    The device PhysX simulates on.
    """

    CPU = "cpu"
    """
    The processor.
    """

    CUDA = "cuda"
    """
    The NVIDIA graphics card.
    """


# %% joints and bodies as Isaac Sim indexes them


@dataclass(frozen=True)
class ArticulatedJoint:
    """
    A joint of one of the stage's articulations, as Isaac Sim indexes it.
    """

    articulation: Articulation
    """
    The articulation the joint belongs to.
    """

    index: int
    """
    The joint's index among the articulation's degrees of freedom.
    """


@dataclass
class JointsOfArticulation:
    """
    Some of the joints of one articulation, as Isaac Sim reads and writes them together.
    """

    articulation: Articulation
    """
    The articulation the joints belong to.
    """

    paths: list[str]
    """
    The paths of the joints' prims.
    """

    indices: list[int]
    """
    The joints' indices among the articulation's degrees of freedom, in the order of
    :attr:`paths`.
    """

    def values_of(self, values: dict[str, float]) -> np.ndarray:
        """
        :param values: A value for each of these joints and possibly others, by their
            paths.
        :return: The values of these joints, in the order of :attr:`paths`, as Isaac Sim
            takes them.
        """
        return np.array([[values[path] for path in self.paths]])


@dataclass(frozen=True)
class BodyPose:
    """
    Where a free body is in the world.
    """

    position: np.ndarray
    """
    The position of the body's origin, in metres.
    """

    orientation: np.ndarray
    """
    The body's orientation as a unit quaternion, in the order w, x, y, z.
    """


# %% pictures


class ReplicatorAnnotator(StrEnum):
    """
    What Isaac Sim's Replicator reads off a camera.
    """

    COLOUR = "rgb"
    """
    The picture, in red, green, blue and alpha.
    """


@dataclass(frozen=True)
class CameraPicture:
    """
    A camera's picture at one size.
    """

    camera_path: str
    """
    The path of the camera's prim.
    """

    height: int
    """
    The picture's height, in pixels.
    """

    width: int
    """
    The picture's width, in pixels.
    """


# %% Isaac Sim's window


class IsaacRenderer(SimulatorRenderer):
    """
    Isaac Sim's window, redrawn on request without stepping the physics again.
    """

    def is_running(self) -> bool:
        return super().is_running() and IsaacApplication().start().is_running()

    def sync(self) -> None:
        RenderingManager.render()


# %% the simulator


@dataclass
class IsaacSimulator(BaseSimulator):
    """
    Steps the physics of a USD stage in Isaac Sim, in steps of :attr:`step_size`, and
    reads and writes its joints and free bodies by the paths of their prims.

    Joint positions and targets are in radians and metres, poses in the world's frame.
    Isaac Sim steps and draws only on the thread that started it, so the simulator is
    started stepped, and stepped from that thread.
    """

    file_path: str = field(default="", kw_only=True)
    """
    The stage to simulate.
    """

    device: PhysicsDevice = field(default=PhysicsDevice.CPU, kw_only=True)
    """
    The device PhysX simulates on.
    """

    _joints: dict[str, ArticulatedJoint] = field(init=False, default_factory=dict)
    """
    Every joint of the stage's articulations, by the path of its prim.
    """

    _bodies: RigidPrim | None = field(init=False, default=None)
    """
    The free bodies whose poses are read and written; ``None`` when there are none.
    """

    _body_paths: list[str] = field(init=False, default_factory=list)
    """
    The paths of the free bodies' prims, in the order of :attr:`_bodies`.
    """

    frames_to_warm_up_a_camera: int = field(default=20, kw_only=True)
    """
    How many times the stage is drawn when a camera is first captured, so its picture is
    there and lit.
    """

    frames_to_draw_a_picture: int = field(default=3, kw_only=True)
    """
    How many times the stage is drawn before a camera's picture shows how the stage is
    now; Replicator hands out the picture of an earlier drawing until then.
    """

    _annotators: dict[CameraPicture, Annotator] = field(
        init=False, default_factory=dict
    )
    """
    What reads each camera's picture, by the camera and the picture's size.
    """

    _drawn_at_step: int | None = field(init=False, default=None)
    """
    The step the stage was last drawn at, so that pictures taken in one step share one
    drawing; ``None`` before the first.
    """

    def start(
        self,
        simulate_in_thread: bool = True,
        render_in_thread: bool = False,
        constraints: SimulatorConstraints | None = None,
        time_out_in_seconds: float = 10.0,
    ) -> None:
        """
        :meth:`BaseSimulator.start`, refusing to step or draw on a thread of its own.

        :raises SteppedOnAnotherThreadError: If a simulation or render thread is asked
            for.
        """
        if simulate_in_thread or render_in_thread:
            raise SteppedOnAnotherThreadError(type(self).__name__)
        super().start(
            simulate_in_thread=False,
            render_in_thread=False,
            constraints=constraints,
            time_out_in_seconds=time_out_in_seconds,
        )

    def start_callback(self) -> None:
        application = IsaacApplication(headless=self.headless).start()
        context = omni.usd.get_context()
        context.open_stage(self.file_path)
        application.update()
        SimulationManager.setup_simulation(dt=self.step_size, device=self.device.value)
        omni.timeline.get_timeline_interface().play()
        application.update()
        for prim in context.get_stage().Traverse():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            articulation = Articulation(str(prim.GetPath()))
            for index, joint_path in enumerate(articulation.dof_paths[0]):
                self._joints[joint_path] = ArticulatedJoint(articulation, index)
        self._renderer = SimulatorRenderer() if self.headless else IsaacRenderer()

    @property
    def joint_paths(self) -> set[str]:
        """
        :return: The paths of every joint of the stage's articulations.
        """
        return set(self._joints)

    def step_callback(self) -> None:
        SimulationManager.step(steps=1)
        self._current_number_of_steps += 1

    def stop(self) -> None:
        omni.timeline.get_timeline_interface().stop()
        super().stop()

    @property
    def current_simulation_time(self) -> float:
        return self.current_number_of_steps * self.step_size

    def track_bodies(self, paths: list[str]) -> None:
        """
        :param paths: The paths of the free bodies whose poses are read and written.
        """
        self._body_paths = list(paths)
        self._bodies = RigidPrim(self._body_paths) if self._body_paths else None

    # %% joints

    def joint_positions(self, paths: Iterable[str]) -> dict[str, float]:
        """
        :param paths: The paths of joints of the stage's articulations.
        :return: The position of each joint, by its path.
        """
        positions = {}
        for joints in self._group_by_articulation(paths):
            values = joints.articulation.get_dof_positions(dof_indices=joints.indices)
            positions.update(zip(joints.paths, map(float, values.numpy()[0])))
        return positions

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """
        Put joints where they are given, without moving them there through the physics.

        :param positions: The position of each joint, by its path.
        """
        for joints in self._group_by_articulation(positions):
            joints.articulation.set_dof_positions(
                joints.values_of(positions), dof_indices=joints.indices
            )

    def set_joint_position_targets(self, targets: dict[str, float]) -> None:
        """
        :param targets: The position each joint's drive drives it towards, by its path.
        """
        for joints in self._group_by_articulation(targets):
            joints.articulation.set_dof_position_targets(
                joints.values_of(targets), dof_indices=joints.indices
            )

    def set_joint_velocity_targets(self, targets: dict[str, float]) -> None:
        """
        :param targets: The velocity each joint's drive drives it at, by its path.
        """
        for joints in self._group_by_articulation(targets):
            joints.articulation.set_dof_velocity_targets(
                joints.values_of(targets), dof_indices=joints.indices
            )

    def _group_by_articulation(
        self, paths: Iterable[str]
    ) -> list[JointsOfArticulation]:
        """
        :param paths: The paths of joints of the stage's articulations.
        :return: The joints, grouped by the articulation they belong to.
        """
        grouped: dict[Articulation, JointsOfArticulation] = {}
        for path in paths:
            joint = self._joints[path]
            joints = grouped.setdefault(
                joint.articulation, JointsOfArticulation(joint.articulation, [], [])
            )
            joints.paths.append(path)
            joints.indices.append(joint.index)
        return list(grouped.values())

    # %% cameras

    def capture_rgb(self, camera_path: str, height: int, width: int) -> np.ndarray:
        """
        :param camera_path: The path of a camera's prim.
        :param height: The picture's height, in pixels.
        :param width: The picture's width, in pixels.
        :return: What the camera sees now, in red, green and blue, one byte each.
        """
        picture = CameraPicture(str(camera_path), height, width)
        if picture not in self._annotators:
            render_product = replicator.create.render_product(
                picture.camera_path, (width, height)
            )
            annotator = replicator.AnnotatorRegistry.get_annotator(
                ReplicatorAnnotator.COLOUR
            )
            annotator.attach([render_product])
            self._annotators[picture] = annotator
            for _ in range(self.frames_to_warm_up_a_camera):
                RenderingManager.render()
        if self._drawn_at_step != self.current_number_of_steps:
            for _ in range(self.frames_to_draw_a_picture):
                RenderingManager.render()
            self._drawn_at_step = self.current_number_of_steps
        return np.asarray(self._annotators[picture].get_data())[:, :, :3]

    # %% free bodies

    def body_poses(self) -> dict[str, BodyPose]:
        """
        :return: The pose of every tracked free body, by the path of its prim.
        """
        if self._bodies is None:
            return {}
        positions, orientations = self._bodies.get_world_poses()
        return {
            path: BodyPose(position, orientation)
            for path, position, orientation in zip(
                self._body_paths, positions.numpy(), orientations.numpy()
            )
        }

    def set_body_poses(self, poses: dict[str, BodyPose]) -> None:
        """
        Put free bodies where they are given, without moving them there through the
        physics.

        :param poses: The pose of each body, by the path of its prim.
        """
        if not poses:
            return
        self._bodies.set_world_poses(
            positions=np.array([pose.position for pose in poses.values()]),
            orientations=np.array([pose.orientation for pose in poses.values()]),
            indices=[self._body_paths.index(path) for path in poses],
        )
