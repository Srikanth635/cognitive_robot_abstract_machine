"""
The stage a world is written as: the names of what the stage holds, and what every
writer of a part of the world shares while the stage is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

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

    ID = "semantic_digital_twin:id"
    """
    The id of the body a prim stands for.
    """

    MEASURED_DEGREE_OF_FREEDOM = "semantic_digital_twin:measured_degree_of_freedom"
    """
    The id of the degree of freedom whose position a joint's position is.
    """

    COMMANDED_DEGREE_OF_FREEDOM = "semantic_digital_twin:commanded_degree_of_freedom"
    """
    The id of the degree of freedom a servo commands a joint through: the measured one
    itself, or, for a wheeled drive's joint, the velocity the drive is commanded with.
    """


class UsdElementName(StrEnum):
    """
    The names the prims of a camera or light take when the world gives it none.
    """

    CAMERA = "camera"
    """
    A camera's.
    """

    LIGHT = "light"
    """
    A light's.
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

    COLLISION_GROUPS = "/World/collisionGroups"
    """
    The scope holding the groups of shapes that do not collide with each other.
    """


class UsdPhysxSchema(StrEnum):
    """
    The PhysX schemas the builder applies by name, readable without PhysX's own USD
    module.
    """

    ARTICULATION = "PhysxArticulationAPI"
    """
    PhysX's settings for an articulation.
    """

    JOINT_AXIS = "PhysxJointAxisAPI"
    """
    PhysX's settings for one axis of a joint, applied under the name of its drive axis.
    """

    MIMIC_JOINT = "PhysxMimicJointAPI"
    """
    PhysX's coupling of one joint's degree of freedom to another's, applied once per
    degree of freedom under the name of its axis.

    .. note::
        Isaac Sim 6.0 marks it deprecated in favour of the Newton schema's mimic API,
        and still reads it.
    """

    FORCE = "PhysxForceAPI"
    """
    A force or acceleration PhysX applies to a rigid body at every step.
    """


class UsdPhysxAttribute(StrEnum):
    """
    The attributes of PhysX schemas the builder writes by name.
    """

    SELF_COLLISIONS = "physxArticulation:enabledSelfCollisions"
    """
    Whether the links of an articulation collide with each other.
    """

    FORCE = "physxForce:force"
    """
    The force, or the acceleration, applied to a rigid body.
    """

    FORCE_MODE = "physxForce:mode"
    """
    Whether :attr:`FORCE` is a force or an acceleration.
    """

    FORCE_IN_WORLD_FRAME = "physxForce:worldFrameEnabled"
    """
    Whether :attr:`FORCE` is stated in the world's frame rather than the body's.
    """


class UsdPhysxForceMode(StrEnum):
    """
    What the vector of :attr:`UsdPhysxAttribute.FORCE` is.
    """

    ACCELERATION = "acceleration"
    """
    An acceleration, applied whatever the body's mass.
    """


class UsdJointAxisProperty(StrEnum):
    """
    The properties of PhysX's joint axis schema, which resist a joint's motion within an
    articulation.
    """

    STATIC_FRICTION_EFFORT = "staticFrictionEffort"
    """
    The largest force or torque friction holds a still joint against.
    """

    DYNAMIC_FRICTION_EFFORT = "dynamicFrictionEffort"
    """
    The force or torque friction resists a moving joint with.
    """

    VISCOUS_FRICTION_COEFFICIENT = "viscousFrictionCoefficient"
    """
    The force or torque friction adds per unit of the joint's speed, per degree for a
    turning joint.
    """

    ARMATURE = "armature"
    """
    The inertia added to the joint, as that of a motor driving it.
    """

    def on(self, instance: str) -> str:
        """
        :param instance: The drive axis the schema is applied under.
        :return: The property's name on that axis.
        """
        return f"physxJointAxis:{instance}:{self.value}"


class UsdMimicJointProperty(StrEnum):
    """
    The properties of PhysX's mimic joint schema, which couples a joint's position to a
    reference joint's by ``position + gearing * reference position + offset = 0``.
    """

    REFERENCE_JOINT = "referenceJoint"
    """
    The joint whose position is mimicked.
    """

    REFERENCE_JOINT_AXIS = "referenceJointAxis"
    """
    The axis of the reference joint's degree of freedom.
    """

    GEARING = "gearing"
    """
    How far the joint moves per unit the reference joint moves, negated.
    """

    OFFSET = "offset"
    """
    Where the joint is when the reference joint is at zero, negated, in degrees or
    metres.
    """

    def on(self, axis: str) -> str:
        """
        :param axis: The axis of the degree of freedom the schema is applied to.
        :return: The property's name on that degree of freedom.
        """
        return f"physxMimicJoint:{axis}:{self.value}"


# %% the stage being written


@dataclass
class WorldStage:
    """
    A stage a world is being written as, and what is known of it so far: the prim each
    body was written as, and the names the prims have taken.

    Every body is a prim directly under :attr:`UsdStagePath.WORLD`, placed where the
    world has it.
    """

    stage: Usd.Stage
    """
    The stage being written.
    """

    world: World
    """
    The world being written.
    """

    body_paths: dict[Body, Sdf.Path] = field(default_factory=dict)
    """
    The prim each body was written as.
    """

    reference_origins: dict[Body, Gf.Matrix4d] = field(default_factory=dict)
    """
    The pose of each body relative to its parent with every joint at zero.
    """

    _taken_names: set[str] = field(default_factory=set)
    """
    The prim names given so far, kept apart from each other.
    """

    def prim_name(self, name: str) -> str:
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

    def joint_path(self, name: str) -> Sdf.Path:
        """
        :param name: The name of what a joint stands for.
        :return: The path of the joint's prim.
        """
        return Sdf.Path(UsdStagePath.JOINTS.value).AppendChild(self.prim_name(name))

    def define_body_prim(self, name: str, pose: Gf.Matrix4d) -> Usd.Prim:
        """
        Write the prim of a body directly under :attr:`UsdStagePath.WORLD`.

        :param name: The body's name.
        :param pose: Where the body is in the world.
        :return: The body's prim.
        """
        path = Sdf.Path(UsdStagePath.WORLD.value).AppendChild(self.prim_name(name))
        xform = UsdGeom.Xform.Define(self.stage, path)
        xform.GetPrim().SetCustomDataByKey(UsdCustomDataKey.NAME.value, name)
        self.set_pose(xform, pose, Gf.Vec3d(1.0, 1.0, 1.0))
        return xform.GetPrim()

    def moves(self, body: Body) -> bool:
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

    @staticmethod
    def world_pose(body: Body) -> Gf.Matrix4d:
        """
        :param body: A body of the world.
        :return: Where the world has the body, as a USD matrix.
        """
        return Gf.Matrix4d(body.global_pose.to_np().T.tolist())

    @staticmethod
    def x_turned_onto(direction: np.ndarray) -> Gf.Rotation:
        """
        :param direction: A direction.
        :return: The rotation that turns the x axis onto it.
        """
        unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        return Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(*unit))

    @staticmethod
    def pose_matrix(position: list[float], quaternion: list[float]) -> Gf.Matrix4d:
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
    def set_pose(
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
