"""
Keeping a world and its simulation in Isaac Sim in step with each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
import omni.usd
from physics_simulators.isaac_simulator import BodyPose, IsaacSimulator
from pxr import Usd
from scipy.spatial.transform import Rotation
from typing_extensions import Any

from semantic_digital_twin.adapters.isaac.exceptions import (
    ServoWithoutJointError,
    WorldStructureChangedUnderIsaacError,
)
from semantic_digital_twin.adapters.multi_sim import (
    MujocoConnectionJoints,
    MultiSimSynchronizer,
)
from semantic_digital_twin.adapters.usd.builder import UsdCustomDataKey
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    WheeledDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_modification import (
    AddActuatorModification,
    AddConnectionModification,
    AddKinematicStructureEntityModification,
)
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    PositionServo,
    VelocityServo,
)

# %% what the stage stands for in the world


@dataclass(frozen=True)
class MeasuringJoint:
    """
    A joint of the stage whose position is a degree of freedom of the world.
    """

    path: str
    """
    The path of the joint's prim.
    """

    degree_of_freedom: DegreeOfFreedom
    """
    The degree of freedom the joint measures.
    """

    multiplier: float
    """
    How far the joint moves per unit of the degree of freedom.
    """

    offset: float
    """
    Where the joint is when the degree of freedom is at zero.
    """

    odometry: bool
    """
    Whether the joint measures a wheeled drive's odometry, which is read back whatever
    drives it and never put where the world has it.
    """


@dataclass(frozen=True)
class FreeBody:
    """
    A body of the stage free to move, standing for a body on a six degree of freedom
    connection of the world.
    """

    path: str
    """
    The path of the body's prim.
    """

    connection: Connection6DoF
    """
    The connection whose degrees of freedom place the body.
    """


# %% the synchronizer


@dataclass(eq=False)
class IsaacSynchronizer(MultiSimSynchronizer):
    """
    Keeps a world and its simulation in Isaac Sim in step: servos are commanded with the
    set points the world gives them, joints no servo drives and free bodies are put where
    the world puts them, and after the physics advances, joints no servo drives, wheeled
    drives' odometry and free bodies are read back into the world. A wheeled drive's
    odometry is only measured: the world puts it only when the simulation starts.

    The world's structure must not change while it is simulated.
    """

    simulator: IsaacSimulator = field(kw_only=True)

    _measuring_joints: list[MeasuringJoint] = field(init=False, default_factory=list)
    """
    Every joint of the stage that measures a degree of freedom of the world.
    """

    _commanded_joints: dict[UUID, str] = field(init=False, default_factory=dict)
    """
    The path of the joint a servo drives, by the id of the degree of freedom the servo
    commands.
    """

    _free_bodies: list[FreeBody] = field(init=False, default_factory=list)
    """
    Every body of the stage free to move.
    """

    _known_positions: dict[UUID, float] = field(init=False, default_factory=dict)
    """
    The position of each degree of freedom the simulation last had, by its id, to tell
    which ones the world has changed since.
    """

    _sent_controls: dict[str, float] = field(init=False, default_factory=dict)
    """
    The control last sent to each servo, by its name.
    """

    _connected: bool = field(init=False, default=False)
    """
    Whether the stage has been read, which needs the simulation started.
    """

    def connect(self) -> None:
        """
        Find what the started simulation's stage stands for in the world, and put the
        simulation where the world is: PhysX starts every articulation with its joints at
        zero.
        """
        prims = list(omni.usd.get_context().get_stage().Traverse())
        self._measuring_joints = self._measuring_joints_on(prims)
        self._commanded_joints = self._commanded_joints_on(prims)
        self._free_bodies = self._free_bodies_on(prims)
        self.simulator.track_bodies([free_body.path for free_body in self._free_bodies])
        self.simulator.set_joint_positions(self._world_joint_positions_on(prims))
        self.simulator.set_body_poses(
            {
                free_body.path: self._body_pose_of(
                    free_body.connection.child.global_pose
                )
                for free_body in self._free_bodies
            }
        )
        self._connected = True
        self.read_back()

    # %% what the stage stands for in the world

    def _measuring_joints_on(self, prims: list[Usd.Prim]) -> list[MeasuringJoint]:
        """
        :param prims: The prims of the stage.
        :return: Every joint among them that measures a degree of freedom of the world.
        """
        degrees_of_freedom = self._degrees_of_freedom_by_id()
        connections = self._connections_by_name()
        odometry = self._drive_degrees_of_freedom()
        measuring_joints = []
        for prim in prims:
            measured = prim.GetCustomDataByKey(
                UsdCustomDataKey.MEASURED_DEGREE_OF_FREEDOM.value
            )
            if measured is None:
                continue
            degree_of_freedom = degrees_of_freedom[measured]
            connection = connections.get(
                prim.GetCustomDataByKey(UsdCustomDataKey.NAME.value)
            )
            is_axis_joint = isinstance(connection, ActiveConnection1DOF)
            measuring_joints.append(
                MeasuringJoint(
                    path=str(prim.GetPath()),
                    degree_of_freedom=degree_of_freedom,
                    multiplier=connection.multiplier if is_axis_joint else 1.0,
                    offset=connection.offset if is_axis_joint else 0.0,
                    odometry=degree_of_freedom.id in odometry,
                )
            )
        return measuring_joints

    def _commanded_joints_on(self, prims: list[Usd.Prim]) -> dict[UUID, str]:
        """
        :param prims: The prims of the stage.
        :return: The path of the joint among them a servo drives, by the id of the
            degree of freedom the servo commands.
        """
        degrees_of_freedom = self._degrees_of_freedom_by_id()
        commanded_joints = {}
        for prim in prims:
            commanded = prim.GetCustomDataByKey(
                UsdCustomDataKey.COMMANDED_DEGREE_OF_FREEDOM.value
            )
            if commanded is None:
                continue
            commanded_joints[degrees_of_freedom[commanded].id] = str(prim.GetPath())
        return commanded_joints

    def _free_bodies_on(self, prims: list[Usd.Prim]) -> list[FreeBody]:
        """
        :param prims: The prims of the stage.
        :return: Every body among them that is free to move.
        """
        bodies = {str(body.id): body for body in self._world.bodies}
        free_bodies = []
        for prim in prims:
            body = bodies.get(prim.GetCustomDataByKey(UsdCustomDataKey.ID.value))
            if body is None or not isinstance(body.parent_connection, Connection6DoF):
                continue
            free_bodies.append(FreeBody(str(prim.GetPath()), body.parent_connection))
        return free_bodies

    def _world_joint_positions_on(self, prims: list[Usd.Prim]) -> dict[str, float]:
        """
        :param prims: The prims of the stage.
        :return: Where the world has each joint among them that stands for one of its
            connections or drive axes, by the path of its prim.
        """
        connections = self._connections_by_name()
        drive_degrees_of_freedom = self._drive_degrees_of_freedom()
        drive_axes = {
            degree_of_freedom.name.name: degree_of_freedom
            for degree_of_freedom in self._world.degrees_of_freedom
            if degree_of_freedom.id in drive_degrees_of_freedom
        }
        positions = {}
        for prim in prims:
            path = str(prim.GetPath())
            if path not in self.simulator.joint_paths:
                continue
            name = prim.GetCustomDataByKey(UsdCustomDataKey.NAME.value)
            if isinstance(connections.get(name), ActiveConnection1DOF):
                positions[path] = float(connections[name].position)
            elif name in drive_axes:
                positions[path] = float(self._world.state[drive_axes[name].id].position)
        return positions

    def _degrees_of_freedom_by_id(self) -> dict[str, DegreeOfFreedom]:
        """
        :return: Every degree of freedom of the world, by its id as the stage writes it.
        """
        return {
            str(degree_of_freedom.id): degree_of_freedom
            for degree_of_freedom in self._world.degrees_of_freedom
        }

    def _connections_by_name(self) -> dict[str, Connection]:
        """
        :return: Every connection of the world, by its name.
        """
        return {
            connection.name.name: connection for connection in self._world.connections
        }

    def _drive_degrees_of_freedom(self) -> set[UUID]:
        """
        :return: The ids of the degrees of freedom of the world's wheeled drives.
        """
        return {
            degree_of_freedom.id
            for connection in self._world.connections
            if isinstance(connection, WheeledDrive)
            for degree_of_freedom in connection.dofs
        }

    # %% world to simulation

    def _on_state_change(self) -> None:
        if not self._connected:
            return
        with self._world._world_lock:
            self._command_servos()
            self._put_where_the_world_has_them()

    def command_actuators_from_world_state(self) -> None:
        """
        Command every servo with the set point the world gives it, whether it changed or
        not.
        """
        self._sent_controls.clear()
        self._command_servos()

    def _command_servos(self) -> None:
        """
        Command every servo whose control changed: a position servo's joint is driven
        towards its set point, a velocity servo's joint at it.
        """
        connections = {
            degree_of_freedom.id: connection
            for connection in self._world.connections
            for degree_of_freedom in connection.dofs
        }
        position_targets = {}
        velocity_targets = {}
        for servo in self._world.actuators:
            if not isinstance(servo, (PositionServo, VelocityServo)):
                continue
            [degree_of_freedom] = servo.dofs
            connection = connections.get(degree_of_freedom.id)
            control = (
                servo.set_point
                if connection is None
                else MujocoConnectionJoints.of(connection, self._world).control_of(
                    servo
                )
            )
            if self._sent_controls.get(servo.name.name) == control:
                continue
            path = self._commanded_joints.get(degree_of_freedom.id)
            if path is None:
                raise ServoWithoutJointError(servo.name.name)
            targets = (
                position_targets
                if isinstance(servo, PositionServo)
                else velocity_targets
            )
            targets[path] = control
            self._sent_controls[servo.name.name] = control
        self.simulator.set_joint_position_targets(position_targets)
        self.simulator.set_joint_velocity_targets(velocity_targets)

    def _put_where_the_world_has_them(self) -> None:
        """
        Put the joints no servo drives, other than odometry, and the free bodies the
        world has moved where the world has them.
        """
        state = self._world.state
        servoed = self._servoed_degrees_of_freedom()
        joint_positions = {}
        for joint in self._measuring_joints:
            degree_of_freedom = joint.degree_of_freedom
            position = state[degree_of_freedom.id].position
            if (
                joint.odometry
                or degree_of_freedom.id in servoed
                or self._known(degree_of_freedom, position)
            ):
                continue
            joint_positions[joint.path] = joint.multiplier * position + joint.offset
            self._known_positions[degree_of_freedom.id] = position
        self.simulator.set_joint_positions(joint_positions)
        body_poses = {}
        for free_body in self._free_bodies:
            moved = False
            for degree_of_freedom in free_body.connection.dofs:
                position = state[degree_of_freedom.id].position
                if not self._known(degree_of_freedom, position):
                    moved = True
                    self._known_positions[degree_of_freedom.id] = position
            if moved:
                body_poses[free_body.path] = self._body_pose_of(
                    free_body.connection.child.global_pose
                )
        self.simulator.set_body_poses(body_poses)

    # %% simulation to world

    def read_back(self) -> None:
        """
        Read the joints no servo drives, wheeled drives' odometry and the free bodies
        back into the world.
        """
        servoed = self._servoed_degrees_of_freedom()
        read_joints = [
            joint
            for joint in self._measuring_joints
            if joint.odometry or joint.degree_of_freedom.id not in servoed
        ]
        positions = self.simulator.joint_positions(
            [joint.path for joint in read_joints]
        )
        poses = self.simulator.body_poses()
        state = self._world.state
        with self._world._world_lock:
            self._state_callback.pause()
            try:
                for joint in read_joints:
                    position = (positions[joint.path] - joint.offset) / joint.multiplier
                    state[joint.degree_of_freedom.id].position = position
                    self._known_positions[joint.degree_of_freedom.id] = position
                for free_body in self._free_bodies:
                    self._place(free_body, poses[free_body.path])
                self._world.notify_state_change()
                self._state_callback.update_previous_world_state()
            finally:
                self._state_callback.resume()

    def _place(self, free_body: FreeBody, pose: BodyPose) -> None:
        """
        Write a free body's pose into the degrees of freedom of its connection.

        :param free_body: The body.
        :param pose: Its pose in the world.
        """
        w, x, y, z = pose.orientation
        world_T_body = HomogeneousTransformationMatrix.from_xyz_quaternion(
            *pose.position, quat_w=w, quat_x=x, quat_y=y, quat_z=z
        ).to_np()
        connection = free_body.connection
        world_T_parent = connection.parent.global_pose.to_np()
        parent_T_connection = connection.parent_T_connection_expression.to_np()
        connection_T_body = (
            np.linalg.inv(world_T_parent @ parent_T_connection) @ world_T_body
        )
        x, y, z = connection_T_body[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(connection_T_body[:3, :3]).as_quat()
        state = self._world.state
        for degree_of_freedom, value in (
            (connection.x, x),
            (connection.y, y),
            (connection.z, z),
            (connection.qx, qx),
            (connection.qy, qy),
            (connection.qz, qz),
            (connection.qw, qw),
        ):
            state[degree_of_freedom.id].position = float(value)
            self._known_positions[degree_of_freedom.id] = float(value)

    # %% servo positions

    def servo_positions(self, servo_names: list[str]) -> dict[str, float]:
        """
        :param servo_names: The names of servos of the world.
        :return: The position of the joint each servo drives, by the servo's name.
        """
        servos = {servo.name.name: servo for servo in self._world.actuators}
        paths = {}
        for servo_name in servo_names:
            [degree_of_freedom] = servos[servo_name].dofs
            paths[servo_name] = self._commanded_joints[degree_of_freedom.id]
        positions = self.simulator.joint_positions(list(paths.values()))
        return {name: positions[path] for name, path in paths.items()}

    # %% structure

    def on_model_change(self, **kwargs: Any) -> None:
        for modification in self._world._model_manager.model_modification_blocks[-1]:
            if isinstance(modification, AddKinematicStructureEntityModification):
                name = modification.kinematic_structure_entity.name.name
            elif isinstance(modification, AddConnectionModification):
                name = modification.connection.name.name
            elif isinstance(modification, AddActuatorModification):
                name = modification.actuator.name.name
            else:
                continue
            raise WorldStructureChangedUnderIsaacError(name)

    def _servoed_degrees_of_freedom(self) -> set[UUID]:
        """
        :return: The ids of the degrees of freedom a servo of the world drives.
        """
        return {
            degree_of_freedom.id
            for servo in self._world.actuators
            for degree_of_freedom in servo.dofs
        }

    @staticmethod
    def _body_pose_of(pose: HomogeneousTransformationMatrix) -> BodyPose:
        """
        :param pose: A body's pose in the world.
        :return: The pose as Isaac Sim takes it.
        """
        x, y, z, w = pose.quaternion.to_np()
        return BodyPose(
            position=pose.position.to_np()[:3], orientation=np.array([w, x, y, z])
        )

    def _known(self, degree_of_freedom: DegreeOfFreedom, position: float) -> bool:
        """
        :return: Whether the simulation already has the degree of freedom at the
            position.
        """
        return self._known_positions.get(degree_of_freedom.id) == position
