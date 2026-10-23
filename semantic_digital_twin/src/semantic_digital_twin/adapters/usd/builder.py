"""
Writing a world as a USD stage with USD Physics, the scene a PhysX simulator such as
Isaac Sim loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, Vt
from typing_extensions import Any, ClassVar, Dict, Union

from semantic_digital_twin.adapters.multi_sim import (
    BodyConverter,
    BoxConverter,
    ConnectionPrismaticConverter,
    ConnectionRevoluteConverter,
    CylinderConverter,
    DriveAxis,
    MeshConverter,
    MujocoConnectionJoints,
    MultiSimBuilder,
    MultiSimCamera,
    MultiSimLight,
    ShapeConverter,
    SphereConverter,
)
from semantic_digital_twin.adapters.usd.exceptions import UnsupportedUsdConnectionError
from semantic_digital_twin.adapters.usd.parser import UsdGeomPrimType
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
    WheeledDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Mesh, Shape
from semantic_digital_twin.world_description.world_entity import (
    Actuator,
    Body,
    Connection,
    PositionServo,
    Region,
    VelocityServo,
)

# %% names on the stage


class UsdCustomDataKey(StrEnum):
    """
    The keys of the custom data the builder leaves on a prim.
    """

    NAME = "semantic_digital_twin:name"
    """
    The name, in the world, of the body, connection or degree of freedom the prim stands
    for.
    """


class UsdStagePath(StrEnum):
    """
    Where on the stage the builder puts what it writes.
    """

    WORLD = "/World"
    """
    The stage's default prim, holding every body.
    """

    PHYSICS_SCENE = "/World/physicsScene"
    """
    The physics scene, which states gravity.
    """

    JOINTS = "/World/joints"
    """
    The scope holding every joint.
    """


# %% converters


@dataclass
class UsdBodyConverter(BodyConverter):
    """
    Converts a body to its mass properties, as USD's mass API states them.
    """

    def _post_convert(
        self, entity: Body, body_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return body_props


@dataclass
class UsdShapeConverter(ShapeConverter):
    """
    Converts a shape to its pose on its body, its color, and the kind of USD geometry
    prim that draws it.
    """

    prim_type: ClassVar[UsdGeomPrimType]
    """
    The kind of USD geometry prim a shape of this kind becomes.
    """

    prim_type_str: str = "prim_type"
    """
    The key for the kind of geometry prim in the output dictionary.
    """

    def _post_convert(
        self, entity: Shape, shape_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        shape_props[self.prim_type_str] = self.prim_type
        return shape_props


@dataclass
class UsdBoxConverter(UsdShapeConverter, BoxConverter):
    prim_type: ClassVar[UsdGeomPrimType] = UsdGeomPrimType.CUBE


@dataclass
class UsdSphereConverter(UsdShapeConverter, SphereConverter):
    prim_type: ClassVar[UsdGeomPrimType] = UsdGeomPrimType.SPHERE


@dataclass
class UsdCylinderConverter(UsdShapeConverter, CylinderConverter):
    prim_type: ClassVar[UsdGeomPrimType] = UsdGeomPrimType.CYLINDER


@dataclass
class UsdMeshConverter(UsdShapeConverter, MeshConverter):
    prim_type: ClassVar[UsdGeomPrimType] = UsdGeomPrimType.MESH


@dataclass
class UsdRevoluteJointConverter(ConnectionRevoluteConverter):
    """
    Converts a revolute connection to the properties of its joint.
    """

    def _post_convert(
        self, entity: RevoluteConnection, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return joint_props


@dataclass
class UsdPrismaticJointConverter(ConnectionPrismaticConverter):
    """
    Converts a prismatic connection to the properties of its joint.
    """

    def _post_convert(
        self, entity: PrismaticConnection, joint_props: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        return joint_props


# %% joints a servo can drive


@dataclass(frozen=True)
class DrivenJoint:
    """
    A joint a servo can drive, as the stage has it.
    """

    path: Sdf.Path
    """
    The joint's prim.
    """

    turns: bool
    """
    Whether the joint turns about its axis rather than slides along it.
    """

    position: float
    """
    Where the world holds the joint, in radians or metres.
    """


# %% the builder


@dataclass
class UsdBuilder(MultiSimBuilder):
    """
    Writes a world as a USD stage with USD Physics.

    Every body is a prim directly under :attr:`UsdStagePath.WORLD`, placed where the
    world has it; a body moves under physics if anything between it and the world's
    root can move, and is a static collider otherwise. Connections become joints between
    those prims, and servos become the drives of their joints.

    .. note::
        Angular quantities on a USD joint are in degrees: its limits, its drive's target,
        and its drive's stiffness and damping, which are per degree.
    """

    drive_link_mass: float = 0.01
    """
    The mass, in kilograms, of each body a wheeled drive's base moves across the floor
    through.
    """

    drive_link_inertia: float = 1e-4
    """
    The moment of inertia, in kilogram square metres, of each of those bodies about each
    of its axes.
    """

    _stage: Usd.Stage = field(init=False)
    """
    The stage being written.
    """

    _body_paths: Dict[Body, Sdf.Path] = field(init=False, default_factory=dict)
    """
    The prim each body was written as.
    """

    _reference_origins: Dict[Body, Gf.Matrix4d] = field(
        init=False, default_factory=dict
    )
    """
    The pose of each body relative to its parent with every joint at zero.
    """

    _driven_joints: Dict[DegreeOfFreedom, DrivenJoint] = field(
        init=False, default_factory=dict
    )
    """
    The joint a servo of each degree of freedom drives.
    """

    _taken_names: set[str] = field(init=False, default_factory=set)
    """
    The prim names given so far, kept apart from each other.
    """

    def _start_build(self, file_path: str):
        self._stage = Usd.Stage.CreateNew(file_path)
        UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self._stage, 1.0)
        world_prim = UsdGeom.Xform.Define(
            self._stage, UsdStagePath.WORLD.value
        ).GetPrim()
        self._stage.SetDefaultPrim(world_prim)
        scene = UsdPhysics.Scene.Define(self._stage, UsdStagePath.PHYSICS_SCENE.value)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)
        UsdGeom.Scope.Define(self._stage, UsdStagePath.JOINTS.value)

    def _end_build(self, file_path: str):
        self._stage.Save()

    def _build_body(self, body: Body):
        prim = self._define_body_prim(body.name.name, self._world_pose(body))
        self._body_paths[body] = prim.GetPath()
        if body.parent_connection is None:
            return
        body_props = UsdBodyConverter.convert(body)
        self._reference_origins[body] = self._pose_matrix(
            body_props[UsdBodyConverter.pos_str], body_props[UsdBodyConverter.quat_str]
        )
        if not self._moves(body):
            return
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if UsdBodyConverter.mass_str not in body_props:
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

    def _build_region(self, region: Region):
        """
        A region has no physics, so the stage leaves it out.
        """

    def _build_shape(
        self,
        parent: Union[Body, Region],
        shape: Shape,
        is_visible: bool,
        is_collidable: bool,
    ):
        if isinstance(parent, Region):
            return
        shape_props = UsdShapeConverter.convert(shape)
        path = self._body_paths[parent].AppendChild(self._prim_name("shape"))
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
        if isinstance(shape, Mesh):
            approximation = (
                UsdPhysics.Tokens.convexHull
                if self._moves(parent)
                else UsdPhysics.Tokens.none
            )
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                approximation
            )

    def _build_connection(self, connection: Connection):
        if isinstance(connection, Connection6DoF):
            return
        if isinstance(connection, FixedConnection):
            self._hold_in_place(connection)
            return
        if isinstance(connection, (RevoluteConnection, PrismaticConnection)):
            self._build_axis_joint(connection)
            return
        if isinstance(connection, WheeledDrive):
            self._build_wheeled_drive(connection)
            return
        raise UnsupportedUsdConnectionError(
            connection_name=connection.name.name,
            connection_type=type(connection).__name__,
        )

    def _build_actuator(self, actuator: Actuator):
        if not isinstance(actuator, (PositionServo, VelocityServo)):
            return
        for degree_of_freedom in actuator.dofs:
            self._build_drive(self._driven_joints[degree_of_freedom], actuator)

    def _build_camera(self, camera: MultiSimCamera):
        """
        The stage leaves cameras to the simulator's own viewer.
        """

    def _build_light(self, light: MultiSimLight):
        """
        The stage leaves lights to the simulator's own viewer.
        """

    # %% joints and drives

    def _hold_in_place(self, connection: Connection) -> None:
        """
        Hold a moving child where the world has it relative to its parent; a child that
        does not move needs no joint.

        :param connection: The connection whose child is held.
        """
        if not self._moves(connection.child):
            return
        joint = UsdPhysics.FixedJoint.Define(
            self._stage, self._joint_path(connection.name.name)
        )
        self._connect(
            joint,
            connection.name.name,
            self._body_paths[connection.parent],
            self._body_paths[connection.child],
        )
        parent_T_child = Gf.Transform(
            self._world_pose(connection.child)
            * self._world_pose(connection.parent).GetInverse()
        )
        joint.CreateLocalPos0Attr(Gf.Vec3f(parent_T_child.GetTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(parent_T_child.GetRotation().GetQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))

    def _build_axis_joint(
        self, connection: Union[RevoluteConnection, PrismaticConnection]
    ) -> None:
        """
        Write a joint that turns about, or slides along, the connection's axis.

        The joint frame on the child is where the connection puts the joint, turned so
        that its x axis is the connection's axis; on the parent it is the same frame with
        the joint at zero, so the joint reads the position the world has.

        :param connection: The connection to write.
        """
        turns = isinstance(connection, RevoluteConnection)
        joint_props = (
            UsdRevoluteJointConverter if turns else UsdPrismaticJointConverter
        ).convert(connection)
        axis = np.asarray(joint_props[UsdRevoluteJointConverter.axis_str], dtype=float)
        alignment = self._x_turned_onto(axis)
        child_T_joint = Gf.Matrix4d().SetTransform(
            alignment,
            Gf.Vec3d(*map(float, joint_props[UsdRevoluteJointConverter.pos_str])),
        )
        joint = self._define_axis_joint(
            name=connection.name.name,
            turns=turns,
            parent_path=self._body_paths[connection.parent],
            parent_T_joint=child_T_joint * self._reference_origins[connection.child],
            child_path=self._body_paths[connection.child],
            child_T_joint=child_T_joint,
        )
        lower, upper = joint_props[UsdRevoluteJointConverter.range_str]
        if lower != upper:
            to_usd = math.degrees if turns else float
            joint.CreateLowerLimitAttr(to_usd(lower))
            joint.CreateUpperLimitAttr(to_usd(upper))
        driven_joint = DrivenJoint(
            path=joint.GetPath(), turns=turns, position=float(connection.position)
        )
        if connection.name.name == connection.raw_dof.name.name:
            self._driven_joints[connection.raw_dof] = driven_joint
        else:
            self._driven_joints.setdefault(connection.raw_dof, driven_joint)

    def _build_wheeled_drive(self, drive: WheeledDrive) -> None:
        """
        Write a wheeled drive the way the MuJoCo simulation has it: if servos drive it,
        its base slides along the x and the y of the drive's ``odom`` and turns about
        its z, through a body of its own between each two axes; otherwise the base is
        held where the world has it.

        :param drive: The drive to write.
        """
        realisation = MujocoConnectionJoints.of(drive, self.world)
        axes: list[DriveAxis] = realisation.axes
        if not axes:
            self._hold_in_place(drive)
            return
        world_T_odom = Gf.Matrix4d(
            drive.parent_T_connection_expression.to_np().T.tolist()
        ) * self._world_pose(drive.parent)
        parent_path = self._body_paths[drive.parent]
        parent_T_odom = Gf.Matrix4d(
            drive.parent_T_connection_expression.to_np().T.tolist()
        )
        travelled = np.zeros(3)
        for index, axis in enumerate(axes):
            alignment = self._x_turned_onto(axis.direction)
            joint_frame = Gf.Matrix4d().SetRotate(alignment)
            position = self.world.state[axis.measured.id].position
            name = realisation.joint_name_of(axis.measured)
            if index == len(axes) - 1:
                child_path = self._body_paths[drive.child]
                child_T_odom = Gf.Matrix4d(
                    np.linalg.inv(
                        drive.connection_T_child_expression.to_np()
                    ).T.tolist()
                )
                child_T_joint = joint_frame * child_T_odom
            else:
                travelled = travelled + position * axis.direction
                link_pose = (
                    Gf.Matrix4d().SetTranslate(Gf.Vec3d(*travelled)) * world_T_odom
                )
                child_path = self._define_drive_link(name, link_pose).GetPath()
                child_T_joint = joint_frame
            joint = self._define_axis_joint(
                name=name,
                turns=axis.turns,
                parent_path=parent_path,
                parent_T_joint=joint_frame * parent_T_odom,
                child_path=child_path,
                child_T_joint=child_T_joint,
            )
            joint.GetPrim().SetCustomDataByKey(
                UsdCustomDataKey.NAME.value, axis.measured.name.name
            )
            if axis.commanded is not None:
                self._driven_joints[axis.commanded] = DrivenJoint(
                    path=joint.GetPath(), turns=axis.turns, position=position
                )
            parent_path = child_path
            parent_T_odom = Gf.Matrix4d(1.0)

    def _define_drive_link(self, name: str, pose: Gf.Matrix4d) -> Usd.Prim:
        """
        Write a light body a wheeled drive's base moves across the floor through.

        :param name: The name of the joint the body is moved by.
        :param pose: Where the body is in the world.
        :return: The body's prim.
        """
        prim = self._define_body_prim(f"{name}_link", pose)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(self.drive_link_mass)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(self.drive_link_inertia))
        return prim

    def _define_axis_joint(
        self,
        name: str,
        turns: bool,
        parent_path: Sdf.Path,
        parent_T_joint: Gf.Matrix4d,
        child_path: Sdf.Path,
        child_T_joint: Gf.Matrix4d,
    ) -> Union[UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint]:
        """
        Write a joint that turns about, or slides along, the x axis of its frame.

        :param name: The name of what the joint stands for.
        :param turns: Whether the joint turns rather than slides.
        :param parent_path: The prim of the body the joint is attached to.
        :param parent_T_joint: The joint's frame on that body, at position zero.
        :param child_path: The prim of the body the joint moves.
        :param child_T_joint: The joint's frame on that body.
        :return: The joint.
        """
        joint_type = UsdPhysics.RevoluteJoint if turns else UsdPhysics.PrismaticJoint
        joint = joint_type.Define(self._stage, self._joint_path(name))
        self._connect(joint, name, parent_path, child_path)
        for frame, position_attribute, rotation_attribute in (
            (parent_T_joint, joint.CreateLocalPos0Attr, joint.CreateLocalRot0Attr),
            (child_T_joint, joint.CreateLocalPos1Attr, joint.CreateLocalRot1Attr),
        ):
            transform = Gf.Transform(frame)
            position_attribute(Gf.Vec3f(transform.GetTranslation()))
            rotation_attribute(Gf.Quatf(transform.GetRotation().GetQuat()))
        joint.CreateAxisAttr(UsdPhysics.Tokens.x)
        return joint

    def _build_drive(
        self, joint: DrivenJoint, servo: Union[PositionServo, VelocityServo]
    ) -> None:
        """
        Drive a joint with a servo's gains: a position servo towards where the world has
        the joint, a velocity servo at the velocity it is commanded with.

        :param joint: The joint.
        :param servo: The servo driving it.
        """
        per_usd_unit = math.pi / 180 if joint.turns else 1.0
        drive = UsdPhysics.DriveAPI.Apply(
            self._stage.GetPrimAtPath(joint.path),
            UsdPhysics.Tokens.angular if joint.turns else UsdPhysics.Tokens.linear,
        )
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateDampingAttr(servo.gains.damping * per_usd_unit)
        drive.CreateMaxForceAttr(servo.gains.torque_limit)
        if isinstance(servo, VelocityServo):
            drive.CreateStiffnessAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            return
        drive.CreateStiffnessAttr(servo.gains.stiffness * per_usd_unit)
        drive.CreateTargetPositionAttr(
            math.degrees(joint.position) if joint.turns else joint.position
        )

    def _connect(
        self,
        joint: UsdPhysics.Joint,
        name: str,
        parent_path: Sdf.Path,
        child_path: Sdf.Path,
    ) -> None:
        """
        Name a joint after what it stands for and attach it to two body prims.

        :param joint: The joint.
        :param name: The name of what it stands for.
        :param parent_path: The prim of the body it is attached to.
        :param child_path: The prim of the body it moves.
        """
        joint.GetPrim().SetCustomDataByKey(UsdCustomDataKey.NAME.value, name)
        joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])

    # %% poses and names

    def _moves(self, body: Body) -> bool:
        """
        :param body: A body of the world.
        :return: Whether anything between the body and the world's root can move.
        """
        connection = body.parent_connection
        while connection is not None:
            if not isinstance(connection, FixedConnection):
                return True
            connection = connection.parent.parent_connection
        return False

    def _define_body_prim(self, name: str, pose: Gf.Matrix4d) -> Usd.Prim:
        """
        Write the prim of a body directly under :attr:`UsdStagePath.WORLD`.

        :param name: The body's name.
        :param pose: Where the body is in the world.
        :return: The body's prim.
        """
        path = Sdf.Path(UsdStagePath.WORLD.value).AppendChild(self._prim_name(name))
        xform = UsdGeom.Xform.Define(self._stage, path)
        xform.GetPrim().SetCustomDataByKey(UsdCustomDataKey.NAME.value, name)
        self._set_pose(xform, pose, Gf.Vec3d(1.0, 1.0, 1.0))
        return xform.GetPrim()

    def _joint_path(self, name: str) -> Sdf.Path:
        """
        :param name: The name of what a joint stands for.
        :return: The path of the joint's prim.
        """
        return Sdf.Path(UsdStagePath.JOINTS.value).AppendChild(self._prim_name(name))

    def _prim_name(self, name: str) -> str:
        """
        :param name: A name from the world.
        :return: A valid prim name for it that no prim has been given yet.
        """
        candidate = Tf.MakeValidIdentifier(name)
        unique = candidate
        suffix = 1
        while unique in self._taken_names:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        self._taken_names.add(unique)
        return unique

    def _define_geometry(
        self, path: Sdf.Path, shape: Shape, shape_props: Dict[str, Any]
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
            geometry = UsdGeom.Cube.Define(self._stage, path)
            geometry.CreateSizeAttr(1.0)
            scale = Gf.Vec3d(*map(float, shape.scale.to_np()[:3]))
        elif prim_type is UsdGeomPrimType.SPHERE:
            geometry = UsdGeom.Sphere.Define(self._stage, path)
            geometry.CreateRadiusAttr(float(shape.radius))
        elif prim_type is UsdGeomPrimType.CYLINDER:
            geometry = UsdGeom.Cylinder.Define(self._stage, path)
            geometry.CreateRadiusAttr(float(shape.radius))
            geometry.CreateHeightAttr(float(shape.height))
            geometry.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            geometry = UsdGeom.Mesh.Define(self._stage, path)
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
        self._set_pose(
            geometry,
            self._pose_matrix(
                shape_props[UsdShapeConverter.pos_str],
                shape_props[UsdShapeConverter.quat_str],
            ),
            scale,
        )
        return geometry

    @staticmethod
    def _world_pose(body: Body) -> Gf.Matrix4d:
        """
        :param body: A body of the world.
        :return: Where the world has the body, as a USD matrix.
        """
        return Gf.Matrix4d(body.global_pose.to_np().T.tolist())

    @staticmethod
    def _x_turned_onto(direction: np.ndarray) -> Gf.Rotation:
        """
        :param direction: A direction.
        :return: The rotation that turns the x axis onto it.
        """
        unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        return Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(*unit))

    @staticmethod
    def _pose_matrix(position: list[float], quaternion: list[float]) -> Gf.Matrix4d:
        """
        :param position: A position.
        :param quaternion: An orientation, as w, x, y and z.
        :return: The pose as a USD matrix.
        """
        w, x, y, z = map(float, quaternion)
        return Gf.Matrix4d().SetTransform(
            Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))),
            Gf.Vec3d(*map(float, position)),
        )

    @staticmethod
    def _set_pose(
        xformable: UsdGeom.Xformable, pose: Gf.Matrix4d, scale: Gf.Vec3d
    ) -> None:
        """
        Place a prim by a translation, an orientation and a scale, the operations PhysX
        reads a rigid body's pose from.

        :param xformable: The prim to place.
        :param pose: Its pose relative to its parent prim.
        :param scale: Its scale.
        """
        transform = Gf.Transform(pose)
        xformable.AddTranslateOp().Set(transform.GetTranslation())
        xformable.AddOrientOp().Set(Gf.Quatf(transform.GetRotation().GetQuat()))
        xformable.AddScaleOp().Set(Gf.Vec3f(scale))
