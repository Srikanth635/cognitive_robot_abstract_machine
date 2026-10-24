"""
Writing a world's bodies onto a stage: where each is, what it weighs, and the shapes it is
drawn and collides with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from krrood.ormatic.utils import classproperty
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
from typing_extensions import Any

from semantic_digital_twin.adapters.multi_sim import (
    BodyConverter,
    BoxConverter,
    CylinderConverter,
    MeshConverter,
    ShapeConverter,
    SphereConverter,
)
from semantic_digital_twin.adapters.usd.contacts import ContactGroupWriter
from semantic_digital_twin.adapters.usd.parser import UsdGeomPrimType
from semantic_digital_twin.adapters.usd.stage import (
    UsdCustomDataKey,
    UsdPhysxAttribute,
    UsdPhysxForceMode,
    UsdPhysxSchema,
    WorldStage,
)
from semantic_digital_twin.world_description.geometry import Mesh, Shape
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    Region,
)

# %% converters


@dataclass
class UsdBodyConverter(BodyConverter):
    """
    Converts a body to its mass properties, as USD's mass API states them.
    """

    def _post_convert(
        self, entity: Body, body_props: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return body_props


@dataclass
class UsdShapeConverter(ShapeConverter):
    """
    Converts a shape to its pose on its body, its color, and the kind of USD geometry
    prim that draws it.
    """

    @classproperty
    def prim_type(cls) -> UsdGeomPrimType:
        """
        :return: The kind of USD geometry prim a shape of this kind becomes.
        """
        raise NotImplementedError

    prim_type_str: str = "prim_type"
    """
    The key for the kind of geometry prim in the output dictionary.
    """

    def _post_convert(
        self, entity: Shape, shape_props: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        shape_props[self.prim_type_str] = self.prim_type
        return shape_props


@dataclass
class UsdBoxConverter(UsdShapeConverter, BoxConverter):
    @classproperty
    def prim_type(cls) -> UsdGeomPrimType:
        return UsdGeomPrimType.CUBE


@dataclass
class UsdSphereConverter(UsdShapeConverter, SphereConverter):
    @classproperty
    def prim_type(cls) -> UsdGeomPrimType:
        return UsdGeomPrimType.SPHERE


@dataclass
class UsdCylinderConverter(UsdShapeConverter, CylinderConverter):
    @classproperty
    def prim_type(cls) -> UsdGeomPrimType:
        return UsdGeomPrimType.CYLINDER


@dataclass
class UsdMeshConverter(UsdShapeConverter, MeshConverter):
    @classproperty
    def prim_type(cls) -> UsdGeomPrimType:
        return UsdGeomPrimType.MESH


# %% settings


@dataclass
class BodySettings:
    """
    How bodies are weighed on the stage.
    """

    minimum_mass: float = 1e-3
    """
    The mass, in kilograms, of a moving body that has neither shapes nor an inertial, so
    that PhysX can move it.
    """

    minimum_inertia: float = 1e-6
    """
    The moment of inertia, in kilogram square metres, of such a body about each of its
    axes.
    """

    gravity: float = 9.81
    """
    The acceleration, in metres per square second, with which gravity pulls every body
    down.
    """


# %% the writer


@dataclass
class BodyWriter:
    """
    Writes each body as a prim placed where the world has it, moving under physics if
    anything between it and the world's root can move and a static collider otherwise,
    with the shapes it is drawn and collides with.
    """

    world_stage: WorldStage
    """
    The stage being written.
    """

    settings: BodySettings
    """
    How bodies are weighed.
    """

    contact_groups: ContactGroupWriter
    """
    What keeps the collision shapes of categories that do not collide apart.
    """

    def build_body(self, body: Body) -> None:
        prim = self.world_stage.define_body_prim(
            body.name.name, self.world_stage.world_pose(body)
        )
        prim.SetCustomDataByKey(UsdCustomDataKey.ID.value, str(body.id))
        self.world_stage.body_paths[body] = prim.GetPath()
        if body.parent_connection is None:
            return
        body_props = UsdBodyConverter.convert(body)
        self.world_stage.reference_origins[body] = self.world_stage.pose_matrix(
            body_props[UsdBodyConverter.pos_str], body_props[UsdBodyConverter.quat_str]
        )
        if not self.world_stage.moves(body):
            return
        UsdPhysics.RigidBodyAPI.Apply(prim)
        self._carry_weight(body, prim)
        if UsdBodyConverter.mass_str not in body_props:
            if not body.collision.shapes:
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(self.settings.minimum_mass)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(self.settings.minimum_inertia))
            return
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(body_props[UsdBodyConverter.mass_str]))
        mass.CreateCenterOfMassAttr(
            Gf.Vec3f(*map(float, body_props[UsdBodyConverter.inertia_pos_str]))
        )
        mass.CreateDiagonalInertiaAttr(
            Gf.Vec3f(*map(float, body_props[UsdBodyConverter.diagonal_inertia_str]))
        )
        w, x, y, z = map(float, body_props[UsdBodyConverter.inertia_quat_str])
        mass.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    def build_shape(
        self,
        parent: Body | Region,
        shape: Shape,
        is_visible: bool,
        is_collidable: bool,
    ):
        if isinstance(parent, Region):
            return
        shape_props = UsdShapeConverter.convert(shape)
        path = self.world_stage.body_paths[parent].AppendChild(
            self.world_stage.prim_name("shape")
        )
        geometry = self._define_geometry(path, shape, shape_props)
        prim = geometry.GetPrim()
        r, g, b, a = shape_props[UsdShapeConverter.rgba_str]
        geometry.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])
        geometry.CreateDisplayOpacityAttr([a])
        if not is_visible:
            geometry.CreatePurposeAttr(UsdGeom.Tokens.guide)
        if not is_collidable:
            return
        UsdPhysics.CollisionAPI.Apply(prim)
        self.contact_groups.add(shape, path)
        if isinstance(shape, Mesh):
            approximation = (
                UsdPhysics.Tokens.convexHull
                if self.world_stage.moves(parent)
                else UsdPhysics.Tokens.none
            )
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                approximation
            )

    def _carry_weight(self, body: Body, prim: Usd.Prim) -> None:
        """
        Carry the part of a moving body's weight that its :class:`GravityCompensation`
        states, as an acceleration up against gravity.

        :param body: A moving body of the world.
        :param prim: The body's prim.
        """
        compensation = body.get_simulator_property_of_type(GravityCompensation)
        if compensation is None:
            return
        prim.AddAppliedSchema(UsdPhysxSchema.FORCE.value)
        prim.CreateAttribute(
            UsdPhysxAttribute.FORCE.value, Sdf.ValueTypeNames.Vector3f, custom=False
        ).Set(Gf.Vec3f(0.0, 0.0, compensation.fraction * self.settings.gravity))
        prim.CreateAttribute(
            UsdPhysxAttribute.FORCE_MODE.value,
            Sdf.ValueTypeNames.Token,
            custom=False,
            variability=Sdf.VariabilityUniform,
        ).Set(UsdPhysxForceMode.ACCELERATION.value)
        prim.CreateAttribute(
            UsdPhysxAttribute.FORCE_IN_WORLD_FRAME.value,
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(True)

    def _define_geometry(
        self, path: Sdf.Path, shape: Shape, shape_props: dict[str, Any]
    ) -> UsdGeom.Gprim:
        """
        Write the geometry prim that draws a shape, placed on its body.

        :param path: Where to write it.
        :param shape: The shape.
        :param shape_props: The shape's converted properties.
        :return: The geometry prim.
        """
        prim_type = shape_props[UsdShapeConverter.prim_type_str]
        scale = Gf.Vec3d(1.0, 1.0, 1.0)
        if prim_type is UsdGeomPrimType.CUBE:
            geometry = UsdGeom.Cube.Define(self.world_stage.stage, path)
            geometry.CreateSizeAttr(1.0)
            scale = Gf.Vec3d(*map(float, shape.scale.to_np()[:3]))
        elif prim_type is UsdGeomPrimType.SPHERE:
            geometry = UsdGeom.Sphere.Define(self.world_stage.stage, path)
            geometry.CreateRadiusAttr(float(shape.radius))
        elif prim_type is UsdGeomPrimType.CYLINDER:
            geometry = UsdGeom.Cylinder.Define(self.world_stage.stage, path)
            geometry.CreateRadiusAttr(float(shape.radius))
            geometry.CreateHeightAttr(float(shape.height))
            geometry.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            geometry = UsdGeom.Mesh.Define(self.world_stage.stage, path)
            mesh = shape.mesh
            geometry.CreatePointsAttr(
                Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32))
            )
            geometry.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32))
            )
            geometry.CreateFaceVertexIndicesAttr(
                Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).flatten())
            )
        self.world_stage.set_pose(
            geometry,
            self.world_stage.pose_matrix(
                shape_props[UsdShapeConverter.pos_str],
                shape_props[UsdShapeConverter.quat_str],
            ),
            scale,
        )
        return geometry
