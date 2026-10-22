"""
Where things are in a scene and where the window looks at it from.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


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
