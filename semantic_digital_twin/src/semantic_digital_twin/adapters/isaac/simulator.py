"""
Stepping the physics of a USD stage in Isaac Sim, and reading and writing its joints and
free bodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from physics_simulators.base_simulator import BaseSimulator
from pxr import UsdPhysics
from scipy.spatial.transform import Rotation
from typing_extensions import ClassVar

from semantic_digital_twin.adapters.isaac.application import IsaacApplication

# %% joints as Isaac Sim indexes them


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


# %% the simulator


@dataclass
class IsaacSimulator(BaseSimulator):
    """
    Steps the physics of a USD stage in Isaac Sim, in steps of :attr:`step_size`, and
    reads and writes its joints and free bodies by the paths of their prims.

    Joint positions and targets are in radians and metres, poses in the world's frame.
    """

    file_path: str = field(default="", kw_only=True)
    """
    The stage to simulate.
    """

    device: str = field(default="cpu", kw_only=True)
    """
    The device PhysX simulates on.
    """

    name: ClassVar[str] = "Isaac Sim"
    ext: ClassVar[str] = "usdc"

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

    def start_callback(self):
        application = IsaacApplication(headless=self.headless)
        application.start()
        context = omni.usd.get_context()
        context.open_stage(self.file_path)
        application.update()
        SimulationManager.setup_simulation(dt=self.step_size, device=self.device)
        omni.timeline.get_timeline_interface().play()
        application.update()
        for prim in context.get_stage().Traverse():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            articulation = Articulation(str(prim.GetPath()))
            for index, joint_path in enumerate(articulation.dof_paths[0]):
                self._joints[joint_path] = ArticulatedJoint(articulation, index)

    @property
    def joint_paths(self) -> set[str]:
        """
        :return: The paths of every joint of the stage's articulations.
        """
        return set(self._joints)

    def step_callback(self):
        self.advance(1)

    def stop(self):
        omni.timeline.get_timeline_interface().stop()
        super().stop()

    @property
    def current_simulation_time(self) -> float:
        return self.current_number_of_steps * self.step_size

    def advance(self, steps: int) -> None:
        """
        Advance the physics.

        :param steps: How many steps of :attr:`step_size` to advance.
        """
        SimulationManager.step(steps=steps)
        self._current_number_of_steps += steps

    def track_bodies(self, paths: list[str]) -> None:
        """
        :param paths: The paths of the free bodies whose poses are read and written.
        """
        self._body_paths = list(paths)
        self._bodies = RigidPrim(self._body_paths) if self._body_paths else None

    # %% joints

    def joint_positions(self, paths: list[str]) -> dict[str, float]:
        """
        :param paths: The paths of joints of the stage's articulations.
        :return: The position of each joint, by its path.
        """
        positions_by_articulation = {}
        positions = {}
        for path in paths:
            joint = self._joints[path]
            if joint.articulation not in positions_by_articulation:
                positions_by_articulation[joint.articulation] = (
                    joint.articulation.get_dof_positions().numpy()[0]
                )
            positions[path] = float(
                positions_by_articulation[joint.articulation][joint.index]
            )
        return positions

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """
        Put joints where they are given, without moving them there through the physics.

        :param positions: The position of each joint, by its path.
        """
        for articulation, indices, values in self._by_articulation(positions):
            articulation.set_dof_positions(values, dof_indices=indices)

    def set_joint_position_targets(self, targets: dict[str, float]) -> None:
        """
        :param targets: The position each joint's drive drives it towards, by its path.
        """
        for articulation, indices, values in self._by_articulation(targets):
            articulation.set_dof_position_targets(values, dof_indices=indices)

    def set_joint_velocity_targets(self, targets: dict[str, float]) -> None:
        """
        :param targets: The velocity each joint's drive drives it at, by its path.
        """
        for articulation, indices, values in self._by_articulation(targets):
            articulation.set_dof_velocity_targets(values, dof_indices=indices)

    def _by_articulation(
        self, values: dict[str, float]
    ) -> list[tuple[Articulation, list[int], np.ndarray]]:
        """
        :param values: A value for each of some joints, by their paths.
        :return: For each articulation those joints belong to: the articulation, the
            joints' indices in it, and their values as Isaac Sim takes them.
        """
        grouped: dict[Articulation, tuple[list[int], list[float]]] = {}
        for path, value in values.items():
            joint = self._joints[path]
            indices, joint_values = grouped.setdefault(joint.articulation, ([], []))
            indices.append(joint.index)
            joint_values.append(value)
        return [
            (articulation, indices, np.array([joint_values]))
            for articulation, (indices, joint_values) in grouped.items()
        ]

    # %% free bodies

    def body_poses(self) -> dict[str, np.ndarray]:
        """
        :return: The pose of every tracked free body in the world, as a 4 by 4 matrix, by
            the path of its prim.
        """
        if self._bodies is None:
            return {}
        positions, orientations = self._bodies.get_world_poses()
        poses = {}
        for path, position, wxyz in zip(
            self._body_paths, positions.numpy(), orientations.numpy()
        ):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
            pose[:3, 3] = position
            poses[path] = pose
        return poses

    def set_body_poses(self, poses: dict[str, np.ndarray]) -> None:
        """
        Put free bodies where they are given, without moving them there through the
        physics.

        :param poses: The pose of each body in the world, as a 4 by 4 matrix, by the
            path of its prim.
        """
        if not poses:
            return
        indices = [self._body_paths.index(path) for path in poses]
        matrices = list(poses.values())
        self._bodies.set_world_poses(
            positions=np.array([matrix[:3, 3] for matrix in matrices]),
            orientations=np.array(
                [
                    Rotation.from_matrix(matrix[:3, :3]).as_quat(scalar_first=True)
                    for matrix in matrices
                ]
            ),
            indices=indices,
        )
