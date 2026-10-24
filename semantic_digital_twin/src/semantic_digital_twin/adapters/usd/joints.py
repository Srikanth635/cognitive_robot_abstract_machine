"""
Writing a world's connections and servos onto a stage: joints between the bodies' prims,
the drives of the joints servos drive, and the articulations moving bodies are joined in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, Sdf, Usd, UsdPhysics
from typing_extensions import Any

from semantic_digital_twin.adapters.multi_sim import (
    ConnectionPrismaticConverter,
    ConnectionRevoluteConverter,
    DriveAxis,
    MujocoConnectionJoints,
)
from semantic_digital_twin.adapters.usd.exceptions import UnsupportedUsdConnectionError
from semantic_digital_twin.adapters.usd.stage import (
    UsdCustomDataKey,
    UsdJointAxisProperty,
    UsdMimicJointProperty,
    UsdPhysxAttribute,
    UsdPhysxSchema,
    WorldStage,
)
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
    WheeledDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    Actuator,
    Body,
    Connection,
    PositionServo,
    VelocityServo,
)

# %% converters


@dataclass
class UsdRevoluteJointConverter(ConnectionRevoluteConverter):
    """
    Converts a revolute connection to the properties of its joint.
    """

    def _post_convert(
        self, entity: RevoluteConnection, joint_props: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return joint_props


@dataclass
class UsdPrismaticJointConverter(ConnectionPrismaticConverter):
    """
    Converts a prismatic connection to the properties of its joint.
    """

    def _post_convert(
        self, entity: PrismaticConnection, joint_props: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return joint_props


# %% joints as the stage has them


@dataclass(frozen=True)
class TurningOrSlidingJoint:
    """
    A joint of the stage that turns about, or slides along, its axis.
    """

    path: Sdf.Path
    """
    The joint's prim.
    """

    turns: bool
    """
    Whether the joint turns about its axis rather than slides along it.
    """

    @property
    def stage_units_per_world_unit(self) -> float:
        """
        :return: How many of the stage's units of the joint's position make one of the
            world's: degrees per radian when it turns, one when it slides.
        """
        return math.degrees(1.0) if self.turns else 1.0

    @property
    def axis_instance(self) -> str:
        """
        :return: The name USD's drive and PhysX's joint axis settings apply under for
            the joint.
        """
        return UsdPhysics.Tokens.angular if self.turns else UsdPhysics.Tokens.linear


@dataclass(frozen=True)
class DrivenJoint(TurningOrSlidingJoint):
    """
    A joint a servo can drive, as the stage has it.
    """

    position: float
    """
    Where the world holds the joint, in radians or metres.
    """

    measured: DegreeOfFreedom
    """
    The degree of freedom whose position the joint's position is.
    """


@dataclass(frozen=True)
class AxisJoint(TurningOrSlidingJoint):
    """
    A joint that turns about or slides along one axis, written for a connection.
    """

    connection: RevoluteConnection | PrismaticConnection
    """
    The connection it was written for.
    """


# %% settings


@dataclass
class JointSettings:
    """
    How the bodies the stage adds for its joints are weighed.
    """

    added_link_mass: float = 0.01
    """
    The mass, in kilograms, of each body the builder adds: those a wheeled drive's base
    moves across the floor through, and the anchor of each articulation.
    """

    added_link_inertia: float = 1e-4
    """
    The moment of inertia, in kilogram square metres, of each of those bodies about each
    of its axes.
    """


# %% the writer


@dataclass
class JointWriter:
    """
    Writes connections as joints between the bodies' prims, and servos as the drives of
    their joints.

    Moving bodies joined by joints, such as a robot or a drawer, are one PhysX
    articulation each, which keeps a chain of light and heavy links steady, is where PhysX
    applies joint friction, and is how Isaac Sim reads and drives joints: it hangs from a
    body of its own, fixed to the static body it stands on by the articulation's root
    joint, so its base is fixed, and its links do not collide with each other.

    .. note::
        Angular quantities on a USD joint are in degrees: its limits, its drive's target,
        and its drive's stiffness and damping, which are per degree.
    """

    world_stage: WorldStage
    """
    The stage being written.
    """

    settings: JointSettings
    """
    How the bodies added for joints are weighed.
    """

    _driven_joints: dict[DegreeOfFreedom, DrivenJoint] = field(
        init=False, default_factory=dict
    )
    """
    The joint a servo of each degree of freedom drives.
    """

    _axis_joints: list[AxisJoint] = field(init=False, default_factory=list)
    """
    Every joint written for a revolute or prismatic connection.
    """

    _articulation_tops: set[Body] = field(init=False, default_factory=set)
    """
    The topmost body of each articulation, the one joined to the static body it stands
    on.
    """

    def __post_init__(self) -> None:
        self._articulation_tops = self._find_articulation_tops()

    def build_connection(self, connection: Connection) -> None:
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

    def build_actuator(self, actuator: Actuator) -> None:
        if not isinstance(actuator, (PositionServo, VelocityServo)):
            return
        for degree_of_freedom in actuator.dofs:
            self._build_drive(self._driven_joints[degree_of_freedom], actuator)

    def finish(self) -> None:
        """
        Couple the joints that share a degree of freedom, and mark each joint a servo
        drives with the degrees of freedom it measures and is commanded through.
        """
        for axis_joint in self._axis_joints:
            self._mimic_shared_degree_of_freedom(axis_joint)
        for commanded, driven_joint in self._driven_joints.items():
            prim = self.world_stage.stage.GetPrimAtPath(driven_joint.path)
            prim.SetCustomDataByKey(
                UsdCustomDataKey.MEASURED_DEGREE_OF_FREEDOM.value,
                str(driven_joint.measured.id),
            )
            prim.SetCustomDataByKey(
                UsdCustomDataKey.COMMANDED_DEGREE_OF_FREEDOM.value, str(commanded.id)
            )

    # %% joints and drives

    def _hold_in_place(self, connection: Connection) -> None:
        """
        Hold a moving child where the world has it relative to its parent; a child that
        does not move needs no joint.

        :param connection: The connection whose child is held.
        """
        if not self.world_stage.moves(connection.child):
            return
        joint = UsdPhysics.FixedJoint.Define(
            self.world_stage.stage, self.world_stage.joint_path(connection.name.name)
        )
        self._connect(
            joint,
            connection.name.name,
            self.world_stage.body_paths[connection.parent],
            self.world_stage.body_paths[connection.child],
        )
        parent_T_child = Gf.Transform(
            self.world_stage.world_pose(connection.child)
            * self.world_stage.world_pose(connection.parent).GetInverse()
        )
        joint.CreateLocalPos0Attr(Gf.Vec3f(parent_T_child.GetTranslation()))
        joint.CreateLocalRot0Attr(Gf.Quatf(parent_T_child.GetRotation().GetQuat()))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))

    def _build_axis_joint(
        self, connection: RevoluteConnection | PrismaticConnection
    ) -> None:
        """
        Write a joint that turns about, or slides along, the connection's axis.

        The joint frame on the child is where the connection puts the joint, turned so
        that its x axis is the connection's axis; on the parent it is the same frame
        with the joint at zero, so the joint reads the position the world has.

        :param connection: The connection to write.
        """
        turns = isinstance(connection, RevoluteConnection)
        joint_props = (
            UsdRevoluteJointConverter if turns else UsdPrismaticJointConverter
        ).convert(connection)
        axis = np.asarray(joint_props[UsdRevoluteJointConverter.axis_str], dtype=float)
        alignment = self.world_stage.x_turned_onto(axis)
        child_T_joint = Gf.Matrix4d().SetTransform(
            alignment,
            Gf.Vec3d(*map(float, joint_props[UsdRevoluteJointConverter.pos_str])),
        )
        joint = self._define_axis_joint(
            name=connection.name.name,
            turns=turns,
            parent_path=self._parent_path(connection),
            parent_T_joint=child_T_joint
            * self.world_stage.reference_origins[connection.child],
            child_path=self.world_stage.body_paths[connection.child],
            child_T_joint=child_T_joint,
        )
        axis_joint = AxisJoint(path=joint.GetPath(), turns=turns, connection=connection)
        self._axis_joints.append(axis_joint)
        if connection.dynamics != JointDynamics():
            self._resist_motion(axis_joint, connection.dynamics)
        lower, upper = joint_props[UsdRevoluteJointConverter.range_str]
        if lower != upper:
            joint.CreateLowerLimitAttr(lower * axis_joint.stage_units_per_world_unit)
            joint.CreateUpperLimitAttr(upper * axis_joint.stage_units_per_world_unit)
        driven_joint = DrivenJoint(
            path=joint.GetPath(),
            turns=turns,
            position=float(connection.position),
            measured=connection.raw_dof,
        )
        if connection.name.name == connection.raw_dof.name.name:
            self._driven_joints[connection.raw_dof] = driven_joint
        else:
            self._driven_joints.setdefault(connection.raw_dof, driven_joint)

    def _resist_motion(self, joint: AxisJoint, dynamics: JointDynamics) -> None:
        """
        Resist a joint's motion as its dynamics have it: dry friction holds it still up
        to a force or torque, damping slows it in proportion to its speed, and armature
        adds inertia.

        :param joint: The joint.
        :param dynamics: The joint's dynamics.
        """
        instance = joint.axis_instance
        prim = self.world_stage.stage.GetPrimAtPath(joint.path)
        prim.AddAppliedSchema(f"{UsdPhysxSchema.JOINT_AXIS.value}:{instance}")
        for prop, value in (
            (UsdJointAxisProperty.STATIC_FRICTION_EFFORT, dynamics.dry_friction),
            (UsdJointAxisProperty.DYNAMIC_FRICTION_EFFORT, dynamics.dry_friction),
            (
                UsdJointAxisProperty.VISCOUS_FRICTION_COEFFICIENT,
                dynamics.damping / joint.stage_units_per_world_unit,
            ),
            (UsdJointAxisProperty.ARMATURE, dynamics.armature),
        ):
            prim.CreateAttribute(
                prop.on(instance), Sdf.ValueTypeNames.Float, custom=False
            ).Set(float(value))

    def _mimic_shared_degree_of_freedom(self, follower: AxisJoint) -> None:
        """
        Couple a joint to the joint a servo of its degree of freedom drives, if it shares
        that degree of freedom with it: the world has each at its own multiple of the
        degree of freedom, plus its own offset. PhysX reads the coupling in the stage's
        units of both joints.

        :param follower: The joint to couple.
        """
        primary_path = self._driven_joints[follower.connection.raw_dof].path
        if follower.path == primary_path:
            return
        [primary] = [
            axis_joint
            for axis_joint in self._axis_joints
            if axis_joint.path == primary_path
        ]
        ratio = follower.connection.multiplier / primary.connection.multiplier
        offset = follower.connection.offset - ratio * primary.connection.offset
        axis = UsdPhysics.Tokens.rotX
        prim = self.world_stage.stage.GetPrimAtPath(follower.path)
        prim.AddAppliedSchema(f"{UsdPhysxSchema.MIMIC_JOINT.value}:{axis}")
        prim.CreateRelationship(
            UsdMimicJointProperty.REFERENCE_JOINT.on(axis), custom=False
        ).SetTargets([primary_path])
        prim.CreateAttribute(
            UsdMimicJointProperty.REFERENCE_JOINT_AXIS.on(axis),
            Sdf.ValueTypeNames.Token,
            custom=False,
            variability=Sdf.VariabilityUniform,
        ).Set(axis)
        prim.CreateAttribute(
            UsdMimicJointProperty.GEARING.on(axis),
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(
            -ratio
            * follower.stage_units_per_world_unit
            / primary.stage_units_per_world_unit
        )
        prim.CreateAttribute(
            UsdMimicJointProperty.OFFSET.on(axis),
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(-offset * follower.stage_units_per_world_unit)

    def _build_wheeled_drive(self, drive: WheeledDrive) -> None:
        """
        Write a wheeled drive the way the MuJoCo simulation has it: if servos drive it,
        its base slides along the x and the y of the drive's ``odom`` and turns about
        its z, through a body of its own between each two axes; otherwise the base is
        held where the world has it.

        :param drive: The drive to write.
        """
        realisation = MujocoConnectionJoints.of(drive, self.world_stage.world)
        axes: list[DriveAxis] = realisation.axes
        if not axes:
            self._hold_in_place(drive)
            return
        world_T_odom = Gf.Matrix4d(
            drive.parent_T_connection_expression.to_np().T.tolist()
        ) * self.world_stage.world_pose(drive.parent)
        parent_path = self._parent_path(drive)
        parent_T_odom = Gf.Matrix4d(
            drive.parent_T_connection_expression.to_np().T.tolist()
        )
        travelled = np.zeros(3)
        for index, axis in enumerate(axes):
            alignment = self.world_stage.x_turned_onto(axis.direction)
            joint_frame = Gf.Matrix4d().SetRotate(alignment)
            position = self.world_stage.world.state[axis.measured.id].position
            name = realisation.joint_name_of(axis.measured)
            if index == len(axes) - 1:
                child_path = self.world_stage.body_paths[drive.child]
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
                    path=joint.GetPath(),
                    turns=axis.turns,
                    position=position,
                    measured=axis.measured,
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
        return self._define_added_link(f"{name}_link", pose)

    def _define_added_link(self, name: str, pose: Gf.Matrix4d) -> Usd.Prim:
        """
        Write a light moving body the world does not have.

        :param name: The body's name.
        :param pose: Where the body is in the world.
        :return: The body's prim.
        """
        prim = self.world_stage.define_body_prim(name, pose)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(self.settings.added_link_mass)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(self.settings.added_link_inertia))
        return prim

    # %% articulations

    def _find_articulation_tops(self) -> set[Body]:
        """
        :return: The topmost body of every group of moving bodies joined by a joint, the
            group standing on a static body.
        """
        joints_below: dict[Body, int] = {}
        for body in self.world_stage.world.bodies:
            if not self.world_stage.moves(body):
                continue
            top = body
            while self.world_stage.moves(
                top.parent_connection.parent
            ) and not isinstance(top.parent_connection, Connection6DoF):
                top = top.parent_connection.parent
            joints_below[top] = joints_below.get(top, 0) + self._joint_count(
                body.parent_connection
            )
        return {
            top
            for top, joints in joints_below.items()
            if joints >= 1 and not isinstance(top.parent_connection, Connection6DoF)
        }

    def _joint_count(self, connection: Connection) -> int:
        """
        :param connection: A connection of the world.
        :return: How many joints that move the stage writes for it.
        """
        if isinstance(connection, (RevoluteConnection, PrismaticConnection)):
            return 1
        if isinstance(connection, WheeledDrive):
            return len(
                MujocoConnectionJoints.of(connection, self.world_stage.world).axes
            )
        return 0

    def _parent_path(self, connection: Connection) -> Sdf.Path:
        """
        :param connection: A connection whose joint is being written.
        :return: The prim the joint is attached to: the parent's, or, for the first
            joint of an articulation, the articulation's anchor.
        """
        if connection.child not in self._articulation_tops:
            return self.world_stage.body_paths[connection.parent]
        return self._define_articulation_anchor(connection.parent, connection.child)

    def _define_articulation_anchor(self, stand: Body, top: Body) -> Sdf.Path:
        """
        Write the body an articulation hangs from, standing where the static body it
        stands on is, and the fixed joint between the two as the articulation's root,
        which fixes the articulation's base. The articulation's topmost body does not
        collide with the static body, as it would not if its joint joined them directly.

        :param stand: The static body the articulation stands on.
        :param top: The articulation's topmost body.
        :return: The anchor's prim.
        """
        pose = self.world_stage.world_pose(stand)
        prim = self._define_added_link(f"{top.name.name}_anchor", pose)
        joint = UsdPhysics.FixedJoint.Define(
            self.world_stage.stage,
            self.world_stage.joint_path(f"{top.name.name}_anchor"),
        )
        joint_prim = joint.GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(joint_prim)
        joint_prim.AddAppliedSchema(UsdPhysxSchema.ARTICULATION.value)
        joint_prim.CreateAttribute(
            UsdPhysxAttribute.SELF_COLLISIONS.value, Sdf.ValueTypeNames.Bool
        ).Set(False)
        self._connect(
            joint,
            f"{top.name.name}_anchor",
            self.world_stage.body_paths[stand],
            prim.GetPath(),
        )
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        UsdPhysics.FilteredPairsAPI.Apply(
            self.world_stage.stage.GetPrimAtPath(self.world_stage.body_paths[top])
        ).CreateFilteredPairsRel().AddTarget(self.world_stage.body_paths[stand])
        return prim.GetPath()

    def _define_axis_joint(
        self,
        name: str,
        turns: bool,
        parent_path: Sdf.Path,
        parent_T_joint: Gf.Matrix4d,
        child_path: Sdf.Path,
        child_T_joint: Gf.Matrix4d,
    ) -> UsdPhysics.RevoluteJoint | UsdPhysics.PrismaticJoint:
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
        joint = joint_type.Define(
            self.world_stage.stage, self.world_stage.joint_path(name)
        )
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
        self, joint: DrivenJoint, servo: PositionServo | VelocityServo
    ) -> None:
        """
        Drive a joint with a servo's gains: a position servo towards where the world has
        the joint, a velocity servo at the velocity it is commanded with.

        :param joint: The joint.
        :param servo: The servo driving it.
        """
        drive = UsdPhysics.DriveAPI.Apply(
            self.world_stage.stage.GetPrimAtPath(joint.path), joint.axis_instance
        )
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateDampingAttr(servo.gains.damping / joint.stage_units_per_world_unit)
        drive.CreateMaxForceAttr(servo.gains.torque_limit)
        if isinstance(servo, VelocityServo):
            drive.CreateStiffnessAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            return
        drive.CreateStiffnessAttr(
            servo.gains.stiffness / joint.stage_units_per_world_unit
        )
        drive.CreateTargetPositionAttr(
            joint.position * joint.stage_units_per_world_unit
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
