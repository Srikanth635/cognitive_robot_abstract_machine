"""
Writing a world as a USD stage with USD Physics, the scene a PhysX simulator such as
Isaac Sim loads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from semantic_digital_twin.adapters.multi_sim import (
    MultiSimBuilder,
    MultiSimCamera,
    MultiSimLight,
)
from semantic_digital_twin.adapters.usd.appearance import (
    AppearanceSettings,
    AppearanceWriter,
)
from semantic_digital_twin.adapters.usd.bodies import BodySettings, BodyWriter
from semantic_digital_twin.adapters.usd.contacts import ContactGroupWriter
from semantic_digital_twin.adapters.usd.joints import JointSettings, JointWriter
from semantic_digital_twin.adapters.usd.stage import UsdStagePath, WorldStage
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.world_entity import (
    Actuator,
    Body,
    Connection,
    Region,
)

# %% the builder


@dataclass
class UsdBuilder(MultiSimBuilder, BodySettings, JointSettings, AppearanceSettings):
    """
    Writes a world as a USD stage with USD Physics, each part of the world by a writer of
    its own: the bodies and their shapes (:class:`BodyWriter`), the collision groups that
    keep contact categories apart (:class:`ContactGroupWriter`), the joints, drives and
    articulations (:class:`JointWriter`), and the cameras and lights
    (:class:`AppearanceWriter`).
    """

    _world_stage: WorldStage = field(init=False)
    """
    The stage being written, and what is known of it so far.
    """

    _bodies: BodyWriter = field(init=False)
    """
    What writes the bodies and their shapes.
    """

    _contact_groups: ContactGroupWriter = field(init=False)
    """
    What keeps the shapes of contact categories that do not collide apart.
    """

    _joints: JointWriter = field(init=False)
    """
    What writes the joints, their drives and the articulations.
    """

    _appearance: AppearanceWriter = field(init=False)
    """
    What writes the cameras and lights.
    """

    def _start_build(self, file_path: str) -> None:
        stage = Usd.Stage.CreateNew(file_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        world_prim = UsdGeom.Xform.Define(stage, UsdStagePath.WORLD.value).GetPrim()
        stage.SetDefaultPrim(world_prim)
        scene = UsdPhysics.Scene.Define(stage, UsdStagePath.PHYSICS_SCENE.value)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(self.gravity)
        UsdGeom.Scope.Define(stage, UsdStagePath.JOINTS.value)
        self._world_stage = WorldStage(stage=stage, world=self.world)
        self._contact_groups = ContactGroupWriter(world_stage=self._world_stage)
        self._bodies = BodyWriter(
            world_stage=self._world_stage,
            settings=self,
            contact_groups=self._contact_groups,
        )
        self._joints = JointWriter(world_stage=self._world_stage, settings=self)
        self._appearance = AppearanceWriter(
            world_stage=self._world_stage, settings=self
        )

    def _end_build(self, file_path: str) -> None:
        self._joints.finish()
        self._contact_groups.write()
        self._world_stage.stage.Save()

    def _build_body(self, body: Body) -> None:
        self._bodies.build_body(body)

    def _build_region(self, region: Region) -> None:
        """
        A region has no physics, so the stage leaves it out.
        """

    def _build_shape(
        self,
        parent: Body | Region,
        shape: Shape,
        is_visible: bool,
        is_collidable: bool,
    ) -> None:
        self._bodies.build_shape(parent, shape, is_visible, is_collidable)

    def _build_connection(self, connection: Connection) -> None:
        self._joints.build_connection(connection)

    def _build_actuator(self, actuator: Actuator) -> None:
        self._joints.build_actuator(actuator)

    def _build_camera(self, camera: MultiSimCamera) -> None:
        self._appearance.build_camera(camera)

    def _build_light(self, light: MultiSimLight) -> None:
        self._appearance.build_light(light)
