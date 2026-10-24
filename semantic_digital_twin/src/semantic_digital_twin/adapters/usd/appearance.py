"""
Writing a world's cameras and lights onto a stage, fixed to their bodies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
from pxr import Gf, UsdGeom, UsdLux

from semantic_digital_twin.adapters.multi_sim import (
    MujocoCamera,
    MujocoLight,
    MultiSimCamera,
    MultiSimLight,
)
from semantic_digital_twin.adapters.usd.exceptions import (
    UnplacedSimulatorElementError,
    UnsupportedFollowingModeError,
)
from semantic_digital_twin.adapters.usd.stage import UsdElementName, WorldStage

# %% settings


@dataclass
class AppearanceSettings:
    """
    How bright the world's lights are, and how near its cameras see, on the stage.
    """

    directional_light_intensity: float = 3000.0
    """
    The intensity of a directional light whose diffuse colour is white; MuJoCo states a
    light's strength as that colour only.
    """

    spot_light_intensity: float = 30000.0
    """
    The intensity of a light at a place whose diffuse colour is white.
    """

    camera_near_clip: float = 0.01
    """
    How near, in metres, a camera sees things.
    """


# %% the writer


@dataclass
class AppearanceWriter:
    """
    Writes the world's cameras and lights under the prims of the bodies they are fixed
    to.
    """

    world_stage: WorldStage
    """
    The stage being written.
    """

    settings: AppearanceSettings
    """
    How bright the lights are and how near the cameras see.
    """

    def build_camera(self, camera: MultiSimCamera) -> None:
        """
        Write a camera fixed to its body; a MuJoCo camera looks along its -z axis with
        its y axis up, as a USD camera does.

        :raises UnplacedSimulatorElementError: If the camera does not say where it is.
        :raises UnsupportedFollowingModeError: If the camera is not fixed to its body.
        """
        if not isinstance(camera, MujocoCamera):
            raise UnplacedSimulatorElementError(camera.name, type(camera).__name__)
        self._refuse_following(camera.name, camera.mode)
        path = self.world_stage.body_paths[camera.body].AppendChild(
            self.world_stage.prim_name(camera.name or UsdElementName.CAMERA)
        )
        usd_camera = UsdGeom.Camera.Define(self.world_stage.stage, path)
        vertical_aperture = usd_camera.GetVerticalApertureAttr().Get()
        usd_camera.CreateFocalLengthAttr(
            vertical_aperture / (2.0 * math.tan(math.radians(camera.fovy) / 2.0))
        )
        usd_camera.CreateClippingRangeAttr(
            Gf.Vec2f(
                self.settings.camera_near_clip,
                usd_camera.GetClippingRangeAttr().Get()[1],
            )
        )
        w, x, y, z = map(float, camera.quaternion)
        UsdGeom.Xformable(usd_camera).AddTransformOp().Set(
            Gf.Matrix4d().SetTransform(
                Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))),
                Gf.Vec3d(*map(float, camera.position)),
            )
        )

    def build_light(self, light: MultiSimLight) -> None:
        """
        Write a light fixed to its body: a directional light as a distant light, any
        other as a point that shines into its cone; its strength is its diffuse colour's
        brightest channel, scaled by :attr:`AppearanceSettings.directional_light_intensity` or
        :attr:`AppearanceSettings.spot_light_intensity`. A light that is off is left out.

        :raises UnplacedSimulatorElementError: If the light does not say where it is.
        :raises UnsupportedFollowingModeError: If the light is not fixed to its body.
        """
        if not isinstance(light, MujocoLight):
            raise UnplacedSimulatorElementError(light.name, type(light).__name__)
        self._refuse_following(light.name, light.mode)
        if not light.active:
            return
        path = self.world_stage.body_paths[light.body].AppendChild(
            self.world_stage.prim_name(light.name or UsdElementName.LIGHT)
        )
        brightness = max(light.diffuse)
        if light.directional:
            usd_light = UsdLux.DistantLight.Define(self.world_stage.stage, path)
            usd_light.CreateIntensityAttr(
                self.settings.directional_light_intensity * brightness
            )
        else:
            usd_light = UsdLux.SphereLight.Define(self.world_stage.stage, path)
            usd_light.CreateTreatAsPointAttr(True)
            usd_light.CreateIntensityAttr(
                self.settings.spot_light_intensity * brightness
            )
            UsdLux.ShapingAPI.Apply(usd_light.GetPrim()).CreateShapingConeAngleAttr(
                float(light.cutoff)
            )
        if brightness > 0:
            usd_light.CreateColorAttr(
                Gf.Vec3f(*(channel / brightness for channel in light.diffuse))
            )
        UsdLux.ShadowAPI.Apply(usd_light.GetPrim()).CreateShadowEnableAttr(
            bool(light.cast_shadow)
        )
        UsdGeom.Xformable(usd_light).AddTransformOp().Set(
            Gf.Matrix4d().SetTransform(
                Gf.Rotation(
                    Gf.Vec3d(0.0, 0.0, -1.0), Gf.Vec3d(*map(float, light.direction))
                ),
                Gf.Vec3d(*map(float, light.position)),
            )
        )

    @staticmethod
    def _refuse_following(element_name: str, mode: mujoco.mjtCamLight) -> None:
        """
        :param element_name: The name of a camera or light.
        :param mode: How it follows its body.
        :raises UnsupportedFollowingModeError: If it is not fixed to its body.
        """
        if mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            raise UnsupportedFollowingModeError(element_name, mode.name)
