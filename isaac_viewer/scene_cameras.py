"""
The cameras and lights a physics viewer scene is watched with under Isaac Sim: cameras
standing in the room, placed relative to where the robot starts, one riding on the robot's
own camera, and the lights the stage needs to be seen at all.

Import it only once Isaac Sim's application runs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, StrEnum

import numpy as np
from isaacsim.core.rendering_manager import ViewportManager
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from semantic_digital_twin.adapters.usd.builder import UsdCustomDataKey, UsdStagePath
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Vector3

# %% names on the stage


class WatchingPrimName(StrEnum):
    """
    The names of the prims the cameras and lights are written as.
    """

    CAMERAS = "Cameras"
    """
    The scope under the world holding the cameras that stand in the room.
    """

    LIGHTS = "Lights"
    """
    The scope under the world holding the lights.
    """

    SKY = "Sky"
    """
    The light from every direction.
    """

    SUN = "Sun"
    """
    The light from one direction, which casts shadows.
    """

    ROBOT_CAMERA = "WatchingCamera"
    """
    The camera under the prim of the robot's own camera body.
    """


# %% lights


@dataclass(frozen=True)
class SceneLighting:
    """
    The lights a stage is watched under; the stage the builder writes has none.
    """

    sky_intensity: float = 1000.0
    """
    How bright the light from every direction is.
    """

    sun_intensity: float = 3000.0
    """
    How bright the light from one direction is.
    """

    sun_tilt: float = 45.0
    """
    How far, in degrees, the sun's light is tilted down from the horizontal.
    """

    sun_heading: float = 30.0
    """
    The direction, in degrees about the vertical, the sun's light comes from.
    """

    def add_to(self, stage: Usd.Stage) -> None:
        """
        :param stage: The stage to light.
        """
        lights = Sdf.Path(UsdStagePath.WORLD).AppendChild(WatchingPrimName.LIGHTS)
        UsdLux.DomeLight.Define(
            stage, lights.AppendChild(WatchingPrimName.SKY)
        ).CreateIntensityAttr(self.sky_intensity)
        sun = UsdLux.DistantLight.Define(
            stage, lights.AppendChild(WatchingPrimName.SUN)
        )
        sun.CreateIntensityAttr(self.sun_intensity)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(
            Gf.Vec3f(self.sun_tilt, 0.0, self.sun_heading)
        )


# %% where a camera stands


@dataclass(frozen=True)
class OffsetFromRobot:
    """
    A point given relative to where the robot's base is when the run starts, along the
    robot's heading.
    """

    forward: float
    """
    How far, in metres, the point is ahead of the robot.
    """

    left: float
    """
    How far, in metres, the point is to the robot's left.
    """

    up: float
    """
    How far, in metres, the point is above the floor under the robot.
    """

    def in_world(self, world_T_robot: np.ndarray) -> np.ndarray:
        """
        :param world_T_robot: The pose of the robot's base in the world.
        :return: The point in the world.
        """
        heading = math.atan2(world_T_robot[1, 0], world_T_robot[0, 0])
        forward = np.array([math.cos(heading), math.sin(heading), 0.0])
        left = np.array([-math.sin(heading), math.cos(heading), 0.0])
        return (
            world_T_robot[:3, 3]
            + self.forward * forward
            + self.left * left
            + np.array([0.0, 0.0, self.up])
        )


class CameraMount(ABC):
    """
    Where a camera is put on the stage, and how it is pointed.
    """

    @abstractmethod
    def mount(self, stage: Usd.Stage, name: str, robot: AbstractRobot) -> Sdf.Path:
        """
        Write the camera onto the stage.

        :param stage: The stage the robot's world is written as.
        :param name: The camera's name, unique among the cameras.
        :param robot: The robot the camera watches.
        :return: The camera's prim.
        """


@dataclass(frozen=True)
class FixedView(CameraMount):
    """
    A camera standing still in the room, looking from one point at another.
    """

    eye: OffsetFromRobot
    """
    Where the camera is.
    """

    target: OffsetFromRobot
    """
    Where the camera looks.
    """

    focal_length: float = 15.0
    """
    The camera's focal length in millimetres; at 15 it sees about 70 degrees across.
    """

    def mount(self, stage: Usd.Stage, name: str, robot: AbstractRobot) -> Sdf.Path:
        path = (
            Sdf.Path(UsdStagePath.WORLD)
            .AppendChild(WatchingPrimName.CAMERAS)
            .AppendChild(name)
        )
        UsdGeom.Camera.Define(stage, path).CreateFocalLengthAttr(self.focal_length)
        world_T_robot = robot.root.global_pose.to_np()
        ViewportManager.set_camera_view(
            str(path),
            eye=list(self.eye.in_world(world_T_robot)),
            target=list(self.target.in_world(world_T_robot)),
        )
        return path


@dataclass(frozen=True)
class RobotCameraView(CameraMount):
    """
    A camera riding on the robot's default camera, looking where it looks, as far across
    as it sees, with the top of the picture up when the run starts.
    """

    def mount(self, stage: Usd.Stage, name: str, robot: AbstractRobot) -> Sdf.Path:
        robot_camera = robot.get_default_camera()
        body_path = self._prim_of_body(stage, str(robot_camera.root.id))
        path = body_path.AppendChild(WatchingPrimName.ROBOT_CAMERA)
        camera = UsdGeom.Camera.Define(stage, path)
        aperture = camera.GetHorizontalApertureAttr().Get()
        camera.CreateFocalLengthAttr(
            aperture
            / (2.0 * math.tan(robot_camera.field_of_view.horizontal_angle / 2.0))
        )
        world_T_body = robot_camera.root.global_pose.to_np()
        looking = robot_camera.forward_facing_axis.to_np()[:3]
        UsdGeom.Xformable(camera).AddTransformOp().Set(
            self._looking_along(
                looking, world_T_body[:3, :3].T @ Vector3.Z().to_np()[:3]
            )
        )
        return path

    @staticmethod
    def _prim_of_body(stage: Usd.Stage, body_id: str) -> Sdf.Path:
        """
        :param stage: The stage the robot's world is written as.
        :param body_id: The id of a body of the world.
        :return: The prim the body is written as.
        """
        [path] = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.GetCustomDataByKey(UsdCustomDataKey.ID.value) == body_id
        ]
        return path

    @staticmethod
    def _looking_along(looking: np.ndarray, up: np.ndarray) -> Gf.Matrix4d:
        """
        :param looking: The direction to look along, in the frame of the body the camera
            rides on.
        :param up: The direction the top of the picture points towards, in that frame.
        :return: The camera's pose on the body; a USD camera looks along its -z axis
            with its y axis up.
        """
        back = -looking / np.linalg.norm(looking)
        top = up - np.dot(up, back) * back
        top = top / np.linalg.norm(top)
        right = np.cross(top, back)
        body_T_camera = np.eye(4)
        body_T_camera[:3, 0] = right
        body_T_camera[:3, 1] = top
        body_T_camera[:3, 2] = back
        return Gf.Matrix4d(body_T_camera.T.tolist())


# %% the cameras a scene is watched with


class SceneCamera(Enum):
    """
    The cameras a scene is watched with, by where they look from; the ones standing in the
    room are placed for a robot that starts facing a counter, a few metres into a kitchen.
    """

    BEHIND_LEFT = FixedView(
        eye=OffsetFromRobot(forward=-0.4, left=1.7, up=2.0),
        target=OffsetFromRobot(forward=0.8, left=-0.1, up=0.6),
    )
    """
    High up behind the robot, to its left.
    """

    BEHIND_RIGHT = FixedView(
        eye=OffsetFromRobot(forward=-0.2, left=-0.9, up=2.1),
        target=OffsetFromRobot(forward=1.0, left=0.1, up=0.8),
    )
    """
    High up behind the robot, to its right.
    """

    ACROSS_THE_COUNTER = FixedView(
        eye=OffsetFromRobot(forward=2.2, left=0.0, up=1.8),
        target=OffsetFromRobot(forward=0.7, left=0.0, up=0.9),
    )
    """
    Beyond the counter, facing the robot.
    """

    ROBOT_CAMERA = RobotCameraView()
    """
    The robot's own view.
    """

    @property
    def label(self) -> str:
        """
        :return: The camera's name as a reader would write it.
        """
        return self.name.replace("_", " ").capitalize()

    def mount(self, stage: Usd.Stage, robot: AbstractRobot) -> Sdf.Path:
        """
        Write the camera onto the stage.

        :param stage: The stage the robot's world is written as.
        :param robot: The robot the camera watches.
        :return: The camera's prim.
        """
        return self.value.mount(stage, self.name.lower(), robot)
