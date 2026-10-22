"""
Where things are in a scene and where the window looks at it from.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
from semantic_digital_twin.spatial_types.spatial_types import Pose
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(frozen=True)
class Position:
    """
    A position in a world, in meters.
    """

    x: float = 0.0
    """
    The distance along the world's x axis, in meters.
    """

    y: float = 0.0
    """
    The distance along the world's y axis, in meters.
    """

    z: float = 0.0
    """
    The distance along the world's z axis, in meters.
    """

    def pose_in(self, world: World, heading: float = 0.0) -> Pose:
        """
        :param world: The world the position is in.
        :param heading: The angle, in radians, the pose is turned by around the vertical
            axis from the world's x axis.
        :return: The pose at this position in the world, upright and turned by the
            heading.
        """
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=heading, reference_frame=world.root
        )


@dataclass(frozen=True)
class ViewerCamera:
    """
    Where the window's camera is first put: the point it looks at, and the direction and
    distance it looks from.
    """

    lookat: Position
    """
    The point the camera looks at.
    """

    distance: float
    """
    How far the camera is from that point, in meters.
    """

    azimuth: float
    """
    The angle, in degrees, of the direction the camera looks along, around the vertical
    axis from the x axis; 90 looks along the y axis, from the side of smaller y.
    """

    elevation: float
    """
    The angle, in degrees, the camera looks down at the point; negative looks down.
    """

    def apply(self, camera: mujoco.MjvCamera) -> None:
        """
        Put a MuJoCo camera where this one is.

        :param camera: The camera of the window.
        """
        camera.lookat[:] = [self.lookat.x, self.lookat.y, self.lookat.z]
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
