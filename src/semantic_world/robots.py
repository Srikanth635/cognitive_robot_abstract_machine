from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

from typing_extensions import Optional, List, Self

from .prefixed_name import PrefixedName
from .spatial_types.spatial_types import Vector3
from .world import World
from .world_entity import Body, RootedView


@dataclass
class RobotView(RootedView):
    """
    Represents a collection of connected robot bodies, starting from a root body, and ending in a unspecified collection
    of tip bodies.
    """
    _robot: AbstractRobot = field(default=None)
    """
    The robot this view belongs to
    """

@dataclass
class KinematicChain(RobotView):
    """
    Represents a kinematic chain in a robot, starting from a root body, and ending in a specific tip body.
    A kinematic chain can contain both manipulators and sensors at the same time, and is not limited to a single
    instance of each.
    """
    tip_body: Body = field(default_factory=Body)
    """
    The tip body of the kinematic chain, which is the last body in the chain.
    """

    manipulator: Optional[Manipulator] = None
    """
    The manipulator of the kinematic chain, if it exists. This is usually a gripper or similar device.
    """

    sensors: List[Sensor] = field(default_factory=list)
    """
    A collection of sensors in the kinematic chain, such as cameras or other sensors.
    """

    @property
    def kinematic_chain(self) -> list[Body]:
        """
        Returns itself as a kinematic chain.
        """
        return self._world.compute_chain_of_bodies(self.root, self.tip_body)


@dataclass
class Manipulator(RobotView):
    """
    Represents a manipulator of a robot. Always has a tool frame.
    """
    tool_frame: Body = field(default_factory=Body)


@dataclass
class Finger(KinematicChain):
    """
    A finger is a kinematic chain, since it should have an unambiguous tip body, and may contain sensors.
    """
    ...


@dataclass
class ParallelGripper(Manipulator):
    """
    Represents a gripper of a robot. Contains a collection of fingers and a thumb. The thumb is a specific finger
    that always needs to touch an object when grasping it, ensuring a stable grasp.
    """
    finger: Finger = field(default_factory=Finger)
    thumb: Finger = field(default=Finger)


@dataclass
class Sensor(RobotView):
    """
    Represents any kind of sensor in a robot.
    """


@dataclass
class FieldOfView:
    """
    Represents the field of view of a camera sensor, defined by the vertical and horizontal angles of the camera's view.
    """
    vertical_angle: float
    horizontal_angle: float


@dataclass
class Camera(Sensor):
    """
    Represents a camera sensor in a robot.
    """
    forward_facing_axis: Vector3 = field(default_factory=Vector3)
    field_of_view: FieldOfView = field(default_factory=FieldOfView)
    minimal_height: float = 0.0
    maximal_height: float = 1.0


@dataclass
class Neck(KinematicChain):
    """
    Represents a special kinematic chain to identify the different bodys of the neck, which is useful to calculate
    for example "LookAt" joint states without needing IK
    """
    roll_body: Optional[Body] = None
    """
    The body which is connected to the connection controlling the roll rotation of the neck.
    """
    pitch_body: Optional[Body] = None
    """
    The body which is connected to the connection controlling the pitch rotation of the neck.
    """
    yaw_body: Optional[Body] = None
    """
    The body which is connected to the connection controlling the yaw rotation of the neck.
    """


@dataclass
class Torso(KinematicChain):
    """
    A Torso is a kinematic chain connecting the base of the robot with a collection of other kinematic chains.
    """
    kinematic_chains: List[KinematicChain] = field(default_factory=list)
    """
    A collection of kinematic chains, such as sensor chains or manipulation chains, that are connected to the torso.
    """


@dataclass
class AbstractRobot(RootedView):
    """
    Specification of an abstract robot. A robot consists of:
    - a root body, which is the base of the robot
    - an optional torso, which is a kinematic chain (usually without a manipulator) connecting the base with a collection
        of other kinematic chains
    - an optional collection of manipulator chains, each containing a manipulator, such as a gripper
    - an optional collection of sensor chains, each containing a sensor, such as a camera
    => If a kinematic chain contains both a manipulator and a sensor, it will be part of both collections
    """
    odom: Body = field(default_factory=Body)
    """
    The odometry body of the robot, which is usually the base footprint.
    """

    torso: Optional[Torso] = None
    """
    The torso of the robot, which is a kinematic chain connecting the base with a collection of other kinematic chains.
    """

    manipulators: List[Manipulator] = field(default_factory=list)
    """
    A collection of manipulators in the robot, such as grippers.
    """

    sensors: List[Sensor] = field(default_factory=list)
    """
    A collection of sensors in the robot, such as cameras.
    """

    manipulator_chains: List[KinematicChain] = field(default_factory=list)
    """
    A collection of all kinematic chains containing a manipulator, such as a gripper.
    """

    sensor_chains: List[KinematicChain] = field(default_factory=list)
    """
    A collection of all kinematic chains containing a sensor, such as a camera.
    """

    @abstractmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a robot view from the given world.
        This method constructs the robot view by identifying and organizing the various semantic components of the robot,
        such as manipulators, sensors, and kinematic chains. It is expected to be implemented in subclasses.

        :param world: The world from which to create the robot view.

        :return: A robot view.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


@dataclass
class PR2(AbstractRobot):
    """
    Represents the Personal Robot 2 (PR2), which was originally created by Willow Garage.
    The PR2 robot consists of two arms, each with a parallel gripper, a head with a camera, and a prismatic torso
    """

    left_arm: KinematicChain = field(default_factory=KinematicChain)
    right_arm: KinematicChain = field(default_factory=KinematicChain)

    def _create_finger(self, world: World, root_name: str, tip_name: str, name: str) -> Finger:
        """
        :param world: The world from which to get the body objects.
        :param root_name: The name of the root body in the world.
        :param tip_name: The name of the tip body in the world.
        :param side: Which side the finger is on, used for naming.

        :return: A Finger object if both bodies are found, otherwise None.
        """
        root = world.get_body_by_name(root_name)
        tip_body = world.get_body_by_name(tip_name)

        finger = Finger(
            root=root,
            tip_body=tip_body,
            name=PrefixedName(name=name, prefix=world.primary_prefix),
            _world=world,
            _robot=self,
        )
        world.add_view(finger)
        return finger

    def _create_parallel_gripper(self, world: World, palm_body_name: str, tool_frame_name: str,
                                finger_bodys: List[Tuple[str, str]], side: str) -> ParallelGripper:
        """
        Creates a ParallelGripper object from the given palm body name, tool frame name, and finger body pairs.

        :param world: The world from which to get the body objects.
        :param palm_body_name: The name of the palm body in the world.
        :param tool_frame_name: The name of the tool frame body in the world.
        :param finger_bodys: A list of tuples containing the root and tip body names for each finger.
        :param side: A side to use for the name of the gripper.
        :return: A ParallelGripper object if the palm and tool frame bodies are found, otherwise None.
        """
        if len(finger_bodys) != 2:
            raise ValueError("Parallel gripper must have exactly two fingers")
        finger, thumb = None, None
        for index, (finger_root_name, finger_tip_name) in enumerate(finger_bodys):
            if index == 0:
                thumb = self._create_finger(world, finger_root_name, finger_tip_name, side + '_gripper_thumb')
            elif index == 1:
                finger = self._create_finger(world, finger_root_name, finger_tip_name, side + '_gripper_finger')

        palm_body = world.get_body_by_name(palm_body_name)
        tool_frame = world.get_body_by_name(tool_frame_name)

        parallel_gripper = ParallelGripper(
            name=PrefixedName(name=side + '_gripper', prefix=world.primary_prefix),
            root=palm_body,
            finger=finger,
            thumb=thumb,
            tool_frame=tool_frame,
            _world=world,
            _robot=self,
        )

        world.add_view(parallel_gripper)
        return parallel_gripper

    def _create_arm(self, world: World, shoulder_body_name: str, gripper: ParallelGripper, side: str) -> KinematicChain:
        """
        Creates a KinematicChain object representing an arm, starting from the shoulder body and ending at the gripper.

        :param world: The world from which to get the body objects.
        :param shoulder_body_name: The name of the shoulder body in the world.
        :param gripper: The Gripper object representing the gripper of the arm.
        :param side: A side to use for the name of the arm.

        :return: A KinematicChain object if the shoulder body and gripper are found, otherwise None.
        """
        shoulder = world.get_body_by_name(shoulder_body_name)
        arm_tip = gripper.root.parent_body

        arm = KinematicChain(
            name=PrefixedName(name=side + '_arm', prefix=world.primary_prefix),
            root=shoulder,
            tip_body=arm_tip,
            manipulator=gripper,
            _world=world,
            _robot=self,
        )
        world.add_view(arm)
        return arm

    def _create_camera(self, world: World) -> Camera:
        """
        Creates a Camera object from the given camera sensor.

        :param world: The world from which to get the body objects.

        :return: A Camera object if the camera body is found, otherwise None.
        """
        camera_body = world.get_body_by_name("wide_stereo_optical_frame")
        camera = Camera(
            name=PrefixedName(name="head_camera", prefix=world.primary_prefix),
            root=camera_body,
            forward_facing_axis=Vector3.from_xyz(0, 0, 1),
            field_of_view=FieldOfView(horizontal_angle=0.99483, vertical_angle=0.75049),
            minimal_height=1.27,
            maximal_height=1.60,
            _world=world,
            _robot=self,
        )
        world.add_view(camera)
        return camera

    def _create_neck(self, world: World, camera: Camera) -> Neck:
        """
        Creates a Neck object from the given camera sensor.

        :param world: The world from which to get the body objects.
        :param camera: The Camera object representing the camera sensor of the neck.

        :return: A Neck object if the camera is found, otherwise None.
        """
        neck_root = world.get_body_by_name("head_pan_link")
        neck_tip_body = world.get_body_by_name("head_tilt_link")

        neck = Neck(
            name=PrefixedName(name='neck', prefix=world.primary_prefix),
            root=neck_root,
            tip_body=neck_tip_body,
            sensors=[camera],
            pitch_body=neck_tip_body,
            yaw_body=neck_root,
            _world=world,
            _robot=self,
        )
        world.add_view(neck)
        return neck

    def _create_torso(self, world: World, manipulator_chains: List[KinematicChain], sensor_chains: List[KinematicChain]) -> Torso:
        """
        Creates a Torso object from the given manipulator and sensor chains.

        :param world: The world from which to get the body objects.
        :param manipulator_chains: A list of KinematicChain objects representing the manipulators of the robot.
        :param sensor_chains: A list of KinematicChain objects representing the sensors of the robot.

        :return: A Torso object if the torso body is found, otherwise None.
        """
        torso_body = world.get_body_by_name("torso_lift_link")
        torso = Torso(
            name=PrefixedName(name='torso', prefix=world.primary_prefix),
            root=torso_body,
            tip_body=torso_body,
            kinematic_chains=manipulator_chains + sensor_chains,
            _world=world,
            _robot=self,
        )
        world.add_view(torso)
        return torso

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a PR2 robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A PR2 robot view.
        """

        robot = cls(
            name=PrefixedName(name='pr2', prefix=world.primary_prefix),
            odom=world.get_body_by_name("odom_combined"),
            root=world.get_body_by_name("base_footprint"),
            _world=world,
        )

        # Create left arm
        left_finger_bodys = [
            ("l_gripper_l_finger_link", "l_gripper_l_finger_tip_link"),
            ("l_gripper_r_finger_link", "l_gripper_r_finger_tip_link"),
        ]

        left_gripper = robot._create_parallel_gripper(world, "l_gripper_palm_link", "l_gripper_tool_frame",
                                                      left_finger_bodys, "left")
        robot.manipulators.append(left_gripper)

        robot.left_arm = robot._create_arm(world, "l_shoulder_pan_link", left_gripper, "left")
        robot.manipulator_chains.append(robot.left_arm)

        # Create right arm
        right_finger_bodys = [
            ("r_gripper_l_finger_link", "r_gripper_l_finger_tip_link"),
            ("r_gripper_r_finger_link", "r_gripper_r_finger_tip_link"),
        ]
        right_gripper = robot._create_parallel_gripper(world, "r_gripper_palm_link", "r_gripper_tool_frame",
                                                       right_finger_bodys, "right")
        robot.manipulators.append(right_gripper)
        robot.right_arm = robot._create_arm(world, "r_shoulder_pan_link", right_gripper, "right")
        robot.manipulator_chains.append(robot.right_arm)

        # Create camera
        camera = robot._create_camera(world)
        robot.sensors.append(camera)

        # Create neck
        neck = robot._create_neck(world, camera)
        robot.sensor_chains.append(neck)

        # Create torso
        torso = robot._create_torso(world, robot.manipulator_chains, robot.sensor_chains)
        robot.torso = torso
        world.add_view(robot)

        return robot


