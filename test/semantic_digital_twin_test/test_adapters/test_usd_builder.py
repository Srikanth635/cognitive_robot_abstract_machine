"""
Tests for writing a world as a USD stage, the scene a PhysX simulator such as Isaac Sim
loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import semantic_digital_twin
from semantic_digital_twin.adapters.multi_sim import ContactCategories, MujocoGeom
from semantic_digital_twin.adapters.usd.builder import (
    UsdBuilder,
    UsdCustomDataKey,
    UsdJointAxisProperty,
    UsdMimicJointProperty,
    UsdPhysxAttribute,
    UsdPhysxForceMode,
    UsdPhysxSchema,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    ServoGains,
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
    VelocityServo,
)

from .usd_stages import PXR_AVAILABLE

if PXR_AVAILABLE:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)

# %% worlds to build


def _box(body: Body, scale: Scale) -> ShapeCollection:
    return ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix(reference_frame=body),
                scale=scale,
            )
        ],
        reference_frame=body,
    )


@dataclass
class HingeWorld:
    """
    A world with a static base, an arm on a servoed hinge, a carriage on a servoed slide
    and a free box, and the handles a test needs.
    """

    world: World
    """
    The world itself.
    """

    base: Body
    """
    The base, fixed to the world's root.
    """

    hinge: RevoluteConnection
    """
    The hinge the arm turns on.
    """

    slide: PrismaticConnection
    """
    The slide the carriage moves along.
    """

    free_box: Body
    """
    A box free to move, one metre above the root.
    """

    hinge_gains: ServoGains
    """
    The gains of the hinge's servo.
    """

    slide_gains: ServoGains
    """
    The gains of the slide's servo.
    """


def _hinge_world(hinge_position: float = 0.0) -> HingeWorld:
    world = World()
    hinge_gains = ServoGains(stiffness=50.0, damping=5.0, torque_limit=2.0)
    slide_gains = ServoGains(stiffness=400.0, damping=40.0, torque_limit=100.0)
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        base = Body(name=PrefixedName("base"))
        base.collision = _box(base, Scale(0.4, 0.4, 0.1))
        world.add_connection(FixedConnection(parent=root, child=base))

        arm = Body(name=PrefixedName("arm"))
        arm.collision = _box(arm, Scale(0.2, 0.02, 0.02))
        hinge_dof = DegreeOfFreedom(
            name=PrefixedName("hinge"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-1.0, upper_position=1.0
            ),
        )
        world.add_degree_of_freedom(hinge_dof)
        hinge = RevoluteConnection(
            name=PrefixedName("hinge"),
            parent=base,
            child=arm,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, z=0.2, reference_frame=base
            ),
            axis=Vector3.Y(reference_frame=arm),
            raw_dof=hinge_dof,
        )
        world.add_connection(hinge)

        carriage = Body(name=PrefixedName("carriage"))
        carriage.collision = _box(carriage, Scale(0.05, 0.05, 0.05))
        slide_dof = DegreeOfFreedom(
            name=PrefixedName("slide"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=0.0, upper_position=0.3
            ),
        )
        world.add_degree_of_freedom(slide_dof)
        slide = PrismaticConnection(
            name=PrefixedName("slide"),
            parent=base,
            child=carriage,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=0.3, reference_frame=base
            ),
            axis=Vector3.X(reference_frame=carriage),
            raw_dof=slide_dof,
        )
        world.add_connection(slide)

        free_box = Body(name=PrefixedName("free.box"))
        free_box.collision = _box(free_box, Scale(0.1, 0.2, 0.3))
        free_box.inertial = Inertial(mass=0.5)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=free_box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=root
                ),
            )
        )

        for name, dof, gains in (
            ("hinge_servo", hinge_dof, hinge_gains),
            ("slide_servo", slide_dof, slide_gains),
        ):
            servo = PositionServo(name=PrefixedName(name), gains=gains)
            servo.add_dof(dof)
            world.add_actuator(servo)
    with world.modify_world():
        world.state[hinge_dof.id].position = hinge_position
    world.notify_state_change()
    return HingeWorld(
        world=world,
        base=base,
        hinge=hinge,
        slide=slide,
        free_box=free_box,
        hinge_gains=hinge_gains,
        slide_gains=slide_gains,
    )


def built(world: World, tmp_path: Path) -> Usd.Stage:
    """
    :return: The stage the builder wrote for the world, opened from its file.
    """
    file_path = tmp_path / "scene.usda"
    UsdBuilder().build_world(world=world, file_path=str(file_path))
    return Usd.Stage.Open(str(file_path))


def prim_of(stage: Usd.Stage, body: Body) -> Usd.Prim:
    """
    :return: The prim the builder made of the body.
    """
    [prim] = [
        prim
        for prim in stage.Traverse()
        if prim.GetCustomDataByKey(UsdCustomDataKey.NAME) == body.name.name
    ]
    return prim


def joint_of(stage: Usd.Stage, connection) -> UsdPhysics.Joint:
    """
    :return: The joint the builder made of the connection.
    """
    [prim] = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        and prim.GetCustomDataByKey(UsdCustomDataKey.NAME) == connection.name.name
    ]
    return UsdPhysics.Joint(prim)


def world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def joint_frame_in_world(stage: Usd.Stage, joint: UsdPhysics.Joint, side: int):
    """
    :return: The rotation matrix and position of the joint's frame in the world, as
        seen from body0 (side 0) or body1 (side 1).
    """
    body_relationship = joint.GetBody0Rel() if side == 0 else joint.GetBody1Rel()
    local_position = (
        joint.GetLocalPos0Attr() if side == 0 else joint.GetLocalPos1Attr()
    ).Get()
    local_rotation = (
        joint.GetLocalRot0Attr() if side == 0 else joint.GetLocalRot1Attr()
    ).Get()
    body = stage.GetPrimAtPath(body_relationship.GetTargets()[0])
    body_to_world = Gf.Transform(world_transform(body))
    body_rotation = np.array(Gf.Matrix3d(body_to_world.GetRotation())).T
    local = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(local_rotation)))).T
    position = np.array(body_to_world.GetTranslation()) + body_rotation @ np.array(
        local_position
    )
    return body_rotation @ local, position


# %% the stage


def test_the_stage_simulates_gravity_pulling_down(tmp_path):
    stage = built(_hinge_world().world, tmp_path)
    [scene] = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    physics_scene = UsdPhysics.Scene(scene)
    assert tuple(physics_scene.GetGravityDirectionAttr().Get()) == (0.0, 0.0, -1.0)
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0


def test_every_body_keeps_its_name_on_its_prim(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    for body in (hinge_world.base, hinge_world.free_box):
        assert prim_of(stage, body).IsValid()


# %% bodies


def test_a_body_is_placed_where_the_world_has_it(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    translation = world_transform(
        prim_of(stage, hinge_world.free_box)
    ).ExtractTranslation()
    expected = hinge_world.free_box.global_pose.to_np()[:3, 3]
    assert list(translation) == pytest.approx(list(expected))


def test_a_free_body_is_a_rigid_body_without_a_joint(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    box = prim_of(stage, hinge_world.free_box)
    assert box.HasAPI(UsdPhysics.RigidBodyAPI)
    joined = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        and box.GetPath() in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
    ]
    assert joined == []


def test_a_body_fixed_to_the_root_does_not_move(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    assert not prim_of(stage, hinge_world.base).HasAPI(UsdPhysics.RigidBodyAPI)


def test_a_body_on_a_joint_moves(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    assert prim_of(stage, hinge_world.hinge.child).HasAPI(UsdPhysics.RigidBodyAPI)


def test_a_body_with_an_inertial_has_its_mass(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    mass = UsdPhysics.MassAPI(prim_of(stage, hinge_world.free_box)).GetMassAttr().Get()
    assert mass == pytest.approx(hinge_world.free_box.inertial.mass)


# %% shapes


def test_a_box_is_a_unit_cube_scaled_to_the_boxs_size(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    [cube] = [
        prim
        for prim in Usd.PrimRange(prim_of(stage, hinge_world.free_box))
        if prim.IsA(UsdGeom.Cube)
    ]
    scale = Gf.Transform(
        world_transform(cube) * world_transform(cube.GetParent()).GetInverse()
    ).GetScale()
    size = UsdGeom.Cube(cube).GetSizeAttr().Get()
    [box] = hinge_world.free_box.collision.shapes
    assert [size * factor for factor in scale] == pytest.approx(
        list(box.scale.to_np()[:3])
    )


def test_a_collision_shape_collides(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    [cube] = [
        prim
        for prim in Usd.PrimRange(prim_of(stage, hinge_world.free_box))
        if prim.IsA(UsdGeom.Cube)
    ]
    assert cube.HasAPI(UsdPhysics.CollisionAPI)


def test_a_mesh_of_a_moving_body_collides_as_its_convex_hull(tmp_path):
    hinge_world = _hinge_world()
    world = hinge_world.world
    milk = Path(semantic_digital_twin.__file__).parents[2] / "resources" / "stl"
    mesh = Mesh(
        filename=str(milk / "milk.stl"),
        origin=HomogeneousTransformationMatrix(reference_frame=hinge_world.free_box),
    )
    with world.modify_world():
        hinge_world.free_box.collision = ShapeCollection(
            [mesh], reference_frame=hinge_world.free_box
        )
    stage = built(world, tmp_path)
    [mesh_prim] = [
        prim
        for prim in Usd.PrimRange(prim_of(stage, hinge_world.free_box))
        if prim.IsA(UsdGeom.Mesh)
    ]
    assert len(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()) == len(mesh.mesh.vertices)
    approximation = UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get()
    assert approximation == UsdPhysics.Tokens.convexHull


# %% joints


def test_a_hinge_becomes_a_revolute_joint_between_its_bodies(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge)
    assert joint.GetPrim().IsA(UsdPhysics.RevoluteJoint)
    [anchor] = joint.GetBody0Rel().GetTargets()
    [anchoring] = [
        UsdPhysics.Joint(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.FixedJoint)
        and anchor in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
    ]
    assert anchoring.GetBody0Rel().GetTargets() == [
        prim_of(stage, hinge_world.hinge.parent).GetPath()
    ]
    assert joint.GetBody1Rel().GetTargets() == [
        prim_of(stage, hinge_world.hinge.child).GetPath()
    ]


def test_a_revolute_joints_limits_are_in_degrees(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    joint = UsdPhysics.RevoluteJoint(joint_of(stage, hinge_world.hinge).GetPrim())
    limits = hinge_world.hinge.dof.limits
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(
        math.degrees(limits.lower.position)
    )
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(
        math.degrees(limits.upper.position)
    )


def test_a_prismatic_joints_limits_are_in_metres(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    joint = UsdPhysics.PrismaticJoint(joint_of(stage, hinge_world.slide).GetPrim())
    limits = hinge_world.slide.dof.limits
    assert joint.GetLowerLimitAttr().Get() == pytest.approx(limits.lower.position)
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(limits.upper.position)


def test_a_joint_turns_about_the_connections_axis(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge)
    rotation, _ = joint_frame_in_world(stage, joint, side=1)
    child_rotation = hinge_world.hinge.child.global_pose.to_np()[:3, :3]
    axis_in_world = child_rotation @ hinge_world.hinge.axis.to_np()[:3]
    assert list(rotation[:, 0]) == pytest.approx(list(axis_in_world), abs=1e-6)


@pytest.mark.parametrize("hinge_position", [0.0, 0.3])
def test_the_joint_holds_the_position_the_world_has(tmp_path, hinge_position):
    hinge_world = _hinge_world(hinge_position)
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge)
    parent_rotation, parent_position = joint_frame_in_world(stage, joint, side=0)
    child_rotation, child_position = joint_frame_in_world(stage, joint, side=1)
    relative = parent_rotation.T @ child_rotation
    angle_about_x = math.atan2(relative[2, 1], relative[1, 1])
    assert list(child_position) == pytest.approx(list(parent_position), abs=1e-6)
    assert list(relative[:, 0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    assert angle_about_x == pytest.approx(hinge_position, abs=1e-6)


def test_a_body_fixed_to_a_moving_body_is_held_by_a_fixed_joint(tmp_path):
    hinge_world = _hinge_world()
    world = hinge_world.world
    with world.modify_world():
        tip = Body(name=PrefixedName("tip"))
        tip.collision = _box(tip, Scale(0.02, 0.02, 0.02))
        world.add_connection(
            FixedConnection(
                parent=hinge_world.hinge.child,
                child=tip,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.1, reference_frame=hinge_world.hinge.child
                ),
            )
        )
    stage = built(world, tmp_path)
    joined = [
        UsdPhysics.Joint(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.FixedJoint)
        and prim_of(stage, tip).GetPath()
        in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
    ]
    assert len(joined) == 1
    assert prim_of(stage, tip).HasAPI(UsdPhysics.RigidBodyAPI)


# %% servos


def test_a_hinges_servo_drives_it_with_its_gains_per_degree(tmp_path):
    hinge_world = _hinge_world(hinge_position=0.3)
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge).GetPrim()
    drive = UsdPhysics.DriveAPI(joint, UsdPhysics.Tokens.angular)
    gains = hinge_world.hinge_gains
    assert drive.GetStiffnessAttr().Get() == pytest.approx(
        gains.stiffness * math.pi / 180
    )
    assert drive.GetDampingAttr().Get() == pytest.approx(gains.damping * math.pi / 180)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(gains.torque_limit)
    assert drive.GetTargetPositionAttr().Get() == pytest.approx(math.degrees(0.3))


def test_a_slides_servo_drives_it_with_its_gains_per_metre(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.slide).GetPrim()
    drive = UsdPhysics.DriveAPI(joint, UsdPhysics.Tokens.linear)
    gains = hinge_world.slide_gains
    assert drive.GetStiffnessAttr().Get() == pytest.approx(gains.stiffness)
    assert drive.GetDampingAttr().Get() == pytest.approx(gains.damping)
    assert drive.GetMaxForceAttr().Get() == pytest.approx(gains.torque_limit)


# %% a base on an omnidirectional drive


@dataclass
class DrivenBaseWorld:
    """
    A world with a base on a servoed omnidirectional drive, standing away from where the
    drive starts.
    """

    world: World
    """
    The world itself.
    """

    drive: OmniDrive
    """
    The drive the base moves across the floor with.
    """

    translation_gains: VelocityServoGains
    """
    The gains of the servos driving the base along its x and y.
    """


def _driven_base_world() -> DrivenBaseWorld:
    world = World()
    translation_gains = VelocityServoGains(damping=200.0, torque_limit=1000.0)
    rotation_gains = VelocityServoGains(damping=50.0, torque_limit=500.0)
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        base = Body(name=PrefixedName("base"))
        base.collision = _box(base, Scale(0.5, 0.5, 0.2))
        drive = OmniDrive.create_with_dofs(
            world=world, parent=root, child=base, name=PrefixedName("odom_T_base")
        )
        world.add_connection(drive)
        for dof, gains in (
            (drive.x_velocity, translation_gains),
            (drive.y_velocity, translation_gains),
            (drive.yaw, rotation_gains),
        ):
            servo = VelocityServo(
                name=PrefixedName(f"{dof.name.name}_servo"), gains=gains
            )
            servo.add_dof(dof)
            world.add_actuator(servo)
    with world.modify_world():
        world.state[drive.x.id].position = 0.5
        world.state[drive.y.id].position = -0.2
        world.state[drive.yaw.id].position = 0.3
    world.notify_state_change()
    return DrivenBaseWorld(
        world=world, drive=drive, translation_gains=translation_gains
    )


def drive_joint_of(stage: Usd.Stage, degree_of_freedom: DegreeOfFreedom):
    """
    :return: The joint the builder made for a degree of freedom of a drive.
    """
    [prim] = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        and prim.GetCustomDataByKey(UsdCustomDataKey.NAME)
        == degree_of_freedom.name.name
    ]
    return UsdPhysics.Joint(prim)


def test_a_servoed_drive_moves_its_base_through_a_slide_a_slide_and_a_hinge(tmp_path):
    driven = _driven_base_world()
    stage = built(driven.world, tmp_path)
    drive = driven.drive
    kinds = [
        drive_joint_of(stage, dof).GetPrim().GetTypeName()
        for dof in (drive.x, drive.y, drive.yaw)
    ]
    assert kinds == [
        "PhysicsPrismaticJoint",
        "PhysicsPrismaticJoint",
        "PhysicsRevoluteJoint",
    ]
    assert drive_joint_of(stage, drive.yaw).GetBody1Rel().GetTargets() == [
        prim_of(stage, drive.child).GetPath()
    ]


def test_the_drives_joints_hold_where_the_world_has_the_base(tmp_path):
    driven = _driven_base_world()
    stage = built(driven.world, tmp_path)
    drive = driven.drive
    readings = []
    for dof in (drive.x, drive.y):
        joint = drive_joint_of(stage, dof)
        parent_rotation, parent_position = joint_frame_in_world(stage, joint, side=0)
        _, child_position = joint_frame_in_world(stage, joint, side=1)
        readings.append(
            float(parent_rotation[:, 0] @ (child_position - parent_position))
        )
    joint = drive_joint_of(stage, drive.yaw)
    parent_rotation, _ = joint_frame_in_world(stage, joint, side=0)
    child_rotation, _ = joint_frame_in_world(stage, joint, side=1)
    relative = parent_rotation.T @ child_rotation
    readings.append(math.atan2(relative[2, 1], relative[1, 1]))
    state = driven.world.state
    assert readings == pytest.approx(
        [
            state[drive.x.id].position,
            state[drive.y.id].position,
            state[drive.yaw.id].position,
        ],
        abs=1e-6,
    )


def test_a_drives_velocity_servo_drives_its_joint_by_velocity(tmp_path):
    driven = _driven_base_world()
    stage = built(driven.world, tmp_path)
    joint = drive_joint_of(stage, driven.drive.x).GetPrim()
    drive_api = UsdPhysics.DriveAPI(joint, UsdPhysics.Tokens.linear)
    assert drive_api.GetStiffnessAttr().Get() == 0.0
    assert drive_api.GetDampingAttr().Get() == pytest.approx(
        driven.translation_gains.damping
    )
    assert drive_api.GetTargetVelocityAttr().Get() == 0.0


# %% articulations


def _arm_chain_world() -> World:
    """
    :return: A world with a static base and an arm of two links on two hinges.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        base = Body(name=PrefixedName("base"))
        base.collision = _box(base, Scale(0.4, 0.4, 0.1))
        world.add_connection(FixedConnection(parent=root, child=base))
        parent = base
        for name in ("upper_arm", "forearm"):
            link = Body(name=PrefixedName(name))
            link.collision = _box(link, Scale(0.2, 0.02, 0.02))
            dof = DegreeOfFreedom(
                name=PrefixedName(f"{name}_joint"),
                limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    lower_position=-1.0, upper_position=1.0
                ),
            )
            world.add_degree_of_freedom(dof)
            world.add_connection(
                RevoluteConnection(
                    name=PrefixedName(f"{name}_joint"),
                    parent=parent,
                    child=link,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.2, reference_frame=parent
                    ),
                    axis=Vector3.Z(reference_frame=link),
                    raw_dof=dof,
                )
            )
            parent = link
    return world


def articulation_roots(stage: Usd.Stage) -> list:
    return [
        prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]


def articulation_anchors(stage: Usd.Stage) -> list:
    """
    :return: The paths of the bodies the articulations' roots hold.
    """
    return [
        target
        for root in articulation_roots(stage)
        for target in UsdPhysics.Joint(root).GetBody1Rel().GetTargets()
    ]


def test_bodies_joined_by_several_joints_are_one_articulation(tmp_path):
    world = _arm_chain_world()
    stage = built(world, tmp_path)
    [root] = articulation_roots(stage)
    base = world.get_body_by_name("base")
    assert UsdPhysics.Joint(root).GetBody0Rel().GetTargets() == [
        prim_of(stage, base).GetPath()
    ]


def test_an_articulation_is_rooted_in_a_fixed_joint_so_its_base_is_fixed(tmp_path):
    stage = built(_arm_chain_world(), tmp_path)
    [root] = articulation_roots(stage)
    assert root.IsA(UsdPhysics.FixedJoint)


def test_the_first_joint_of_an_articulation_hangs_from_its_anchor(tmp_path):
    world = _arm_chain_world()
    stage = built(world, tmp_path)
    [anchor] = articulation_anchors(stage)
    first_joint = joint_of(stage, world.get_connection_by_name("upper_arm_joint"))
    assert first_joint.GetBody0Rel().GetTargets() == [anchor]


def test_an_articulations_anchor_stands_where_its_static_parent_is(tmp_path):
    world = _arm_chain_world()
    stage = built(world, tmp_path)
    [anchor] = articulation_anchors(stage)
    base = prim_of(stage, world.get_body_by_name("base"))
    assert world_transform(stage.GetPrimAtPath(anchor)) == world_transform(base)


def test_an_articulation_does_not_collide_with_itself(tmp_path):
    stage = built(_arm_chain_world(), tmp_path)
    [root] = articulation_roots(stage)
    assert root.GetAttribute(UsdPhysxAttribute.SELF_COLLISIONS).Get() is False


def test_a_body_on_a_single_joint_is_an_articulation_of_its_own(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    anchors = articulation_anchors(stage)
    for connection in (hinge_world.hinge, hinge_world.slide):
        [body0] = joint_of(stage, connection).GetBody0Rel().GetTargets()
        assert body0 in anchors
    assert len(anchors) == 2


def test_an_articulations_top_body_does_not_collide_with_the_body_it_stands_on(
    tmp_path,
):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    arm = prim_of(stage, hinge_world.hinge.child)
    base = prim_of(stage, hinge_world.hinge.parent)
    filtered = UsdPhysics.FilteredPairsAPI(arm).GetFilteredPairsRel().GetTargets()
    assert filtered == [base.GetPath()]


def test_a_servoed_drive_with_its_base_is_one_articulation(tmp_path):
    driven = _driven_base_world()
    stage = built(driven.world, tmp_path)
    [anchor] = articulation_anchors(stage)
    first_joint = drive_joint_of(stage, driven.drive.x)
    assert first_joint.GetBody0Rel().GetTargets() == [anchor]


# %% joints that follow another


@dataclass
class FollowingHingesWorld:
    """
    A world with a hinge and two hinges following it through the one degree of freedom
    they share.
    """

    world: World
    """
    The world itself.
    """

    hinge: RevoluteConnection
    """
    The hinge named after the shared degree of freedom.
    """

    mirrored: RevoluteConnection
    """
    A hinge turning the other way.
    """

    geared: RevoluteConnection
    """
    A hinge turning half as far, and offset.
    """


def _following_hinges_world() -> FollowingHingesWorld:
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        base = Body(name=PrefixedName("base"))
        base.collision = _box(base, Scale(0.4, 0.4, 0.1))
        world.add_connection(FixedConnection(parent=root, child=base))
        dof = DegreeOfFreedom(
            name=PrefixedName("hinge"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-1.0, upper_position=1.0
            ),
        )
        world.add_degree_of_freedom(dof)
        hinges = {}
        for index, (name, multiplier, offset) in enumerate(
            (("hinge", 1.0, 0.0), ("mirrored", -1.0, 0.0), ("geared", 0.5, 0.1))
        ):
            arm = Body(name=PrefixedName(f"{name}_arm"))
            arm.collision = _box(arm, Scale(0.2, 0.02, 0.02))
            hinges[name] = RevoluteConnection(
                name=PrefixedName(name),
                parent=base,
                child=arm,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.1 * (index + 1), reference_frame=base
                ),
                axis=Vector3.Z(reference_frame=arm),
                raw_dof=dof,
                multiplier=multiplier,
                offset=offset,
            )
            world.add_connection(hinges[name])
    return FollowingHingesWorld(
        world=world,
        hinge=hinges["hinge"],
        mirrored=hinges["mirrored"],
        geared=hinges["geared"],
    )


@dataclass
class InsetDoorWorld:
    """
    A world with a door inset in a cabinet: as the door turns open on the hinge at its
    edge, the shapeless frame the hinge is on slides out of the cabinet with it.
    """

    world: World
    """
    The world itself.
    """

    hinge: RevoluteConnection
    """
    The joint the door turns on, named after the degree of freedom it shares with the
    frame's joint.
    """

    frame_joint: PrismaticConnection
    """
    The joint the frame slides out of the cabinet along, by a fraction of the door's
    turn.
    """


def _inset_door_world() -> InsetDoorWorld:
    world = World()
    dynamics = JointDynamics(armature=0.0, dry_friction=1.0, damping=0.5)
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        cabinet = Body(name=PrefixedName("cabinet"))
        cabinet.collision = _box(cabinet, Scale(0.5, 0.5, 0.8))
        world.add_connection(
            FixedConnection(
                parent=root,
                child=cabinet,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.4, reference_frame=root
                ),
            )
        )
        opening = DegreeOfFreedom(
            name=PrefixedName("door_joint"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=0.0, upper_position=math.pi / 2
            ),
        )
        world.add_degree_of_freedom(opening)
        frame = Body(name=PrefixedName("door_frame"))
        frame_joint = PrismaticConnection(
            name=PrefixedName("door_frame_joint"),
            parent=cabinet,
            child=frame,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.262, y=-0.2, reference_frame=cabinet
            ),
            axis=Vector3.X(reference_frame=frame),
            raw_dof=opening,
            multiplier=0.01,
            dynamics=dynamics,
        )
        world.add_connection(frame_joint)
        door = Body(name=PrefixedName("door"))
        door.collision = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=0.2, reference_frame=door
                    ),
                    scale=Scale(0.02, 0.4, 0.6),
                )
            ],
            reference_frame=door,
        )
        hinge = RevoluteConnection(
            name=PrefixedName("door_joint"),
            parent=frame,
            child=door,
            axis=-Vector3.Z(reference_frame=door),
            raw_dof=opening,
            dynamics=dynamics,
        )
        world.add_connection(hinge)
    return InsetDoorWorld(world=world, hinge=hinge, frame_joint=frame_joint)


def authored_api_schemas(prim: Usd.Prim) -> list[str]:
    """
    :return: The API schemas authored on the prim, including those of plugins usd-core
        does not have, such as PhysX's.
    """
    list_op = prim.GetMetadata("apiSchemas")
    return [] if list_op is None else list(list_op.GetAddedOrExplicitItems())


def mimicking(joint: UsdPhysics.Joint):
    """
    :return: The PhysX mimic relationship authored on the joint's turning degree of
        freedom.
    """
    prim = joint.GetPrim()
    axis = UsdPhysics.Tokens.rotX
    return (
        prim.GetRelationship(
            UsdMimicJointProperty.REFERENCE_JOINT.on(axis)
        ).GetTargets(),
        prim.GetAttribute(UsdMimicJointProperty.GEARING.on(axis)).Get(),
        prim.GetAttribute(UsdMimicJointProperty.OFFSET.on(axis)).Get(),
    )


def test_a_joint_following_another_mimics_it(tmp_path):
    following = _following_hinges_world()
    stage = built(following.world, tmp_path)
    joint = joint_of(stage, following.mirrored)
    assert f"{UsdPhysxSchema.MIMIC_JOINT}:{UsdPhysics.Tokens.rotX}" in (
        authored_api_schemas(joint.GetPrim())
    )
    references, _, _ = mimicking(joint)
    assert references == [joint_of(stage, following.hinge).GetPath()]


def test_the_joint_named_after_the_shared_degree_of_freedom_mimics_nothing(tmp_path):
    following = _following_hinges_world()
    stage = built(following.world, tmp_path)
    assert authored_api_schemas(joint_of(stage, following.hinge).GetPrim()) == []


@pytest.mark.parametrize("name", ["mirrored", "geared"])
def test_a_mimicking_joint_turns_by_its_multiplier_and_offset(tmp_path, name):
    following = _following_hinges_world()
    stage = built(following.world, tmp_path)
    follower = getattr(following, name)
    _, gearing, offset = mimicking(joint_of(stage, follower))
    reference = 0.4
    position = -gearing * reference - math.radians(offset)
    assert position == pytest.approx(
        follower.multiplier * reference + follower.offset, abs=1e-6
    )


def test_a_sliding_joint_following_a_turning_one_slides_by_its_multiplier_per_radian(
    tmp_path,
):
    inset_door = _inset_door_world()
    stage = built(inset_door.world, tmp_path)
    frame_joint = inset_door.frame_joint
    _, gearing, offset = mimicking(joint_of(stage, frame_joint))
    reference = 0.4
    position = -gearing * math.degrees(reference) - offset
    assert position == pytest.approx(
        frame_joint.multiplier * reference + frame_joint.offset, abs=1e-6
    )


# %% joint friction


def _hinge_world_with_dynamics() -> HingeWorld:
    hinge_world = _hinge_world()
    with hinge_world.world.modify_world():
        hinge_world.hinge.dynamics = JointDynamics(
            armature=0.05, dry_friction=1.0, damping=0.5
        )
    return hinge_world


def joint_axis_value(joint: UsdPhysics.Joint, instance: str, prop) -> float:
    return joint.GetPrim().GetAttribute(prop.on(instance)).Get()


def test_a_joints_dry_friction_resists_it_as_physx_friction_effort(tmp_path):
    hinge_world = _hinge_world_with_dynamics()
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge)
    dry_friction = hinge_world.hinge.dynamics.dry_friction
    for prop in (
        UsdJointAxisProperty.STATIC_FRICTION_EFFORT,
        UsdJointAxisProperty.DYNAMIC_FRICTION_EFFORT,
    ):
        assert joint_axis_value(
            joint, UsdPhysics.Tokens.angular, prop
        ) == pytest.approx(dry_friction)


def test_a_turning_joints_damping_is_viscous_friction_per_degree(tmp_path):
    hinge_world = _hinge_world_with_dynamics()
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge)
    viscous = joint_axis_value(
        joint,
        UsdPhysics.Tokens.angular,
        UsdJointAxisProperty.VISCOUS_FRICTION_COEFFICIENT,
    )
    assert viscous == pytest.approx(hinge_world.hinge.dynamics.damping * math.pi / 180)


def test_a_joints_armature_carries_over(tmp_path):
    hinge_world = _hinge_world_with_dynamics()
    stage = built(hinge_world.world, tmp_path)
    joint = joint_of(stage, hinge_world.hinge)
    armature = joint_axis_value(
        joint, UsdPhysics.Tokens.angular, UsdJointAxisProperty.ARMATURE
    )
    assert armature == pytest.approx(hinge_world.hinge.dynamics.armature)


def test_a_joint_without_dynamics_gets_no_physx_joint_axis(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    schemas = authored_api_schemas(joint_of(stage, hinge_world.hinge).GetPrim())
    assert not any(schema.startswith(UsdPhysxSchema.JOINT_AXIS) for schema in schemas)


def test_a_single_joint_with_friction_is_an_articulation(tmp_path):
    hinge_world = _hinge_world_with_dynamics()
    stage = built(hinge_world.world, tmp_path)
    [body0] = joint_of(stage, hinge_world.hinge).GetBody0Rel().GetTargets()
    assert body0 in articulation_anchors(stage)


# %% contact categories


def _furniture_world() -> World:
    """
    :return: A world whose two cabinets are in a contact category of their own that does
        not collide with itself, next to a free box in the default category.
    """
    hinge_world = _hinge_world()
    world = hinge_world.world
    with world.modify_world():
        for name in ("cabinet", "drawer"):
            body = Body(name=PrefixedName(name))
            body.collision = _box(body, Scale(0.3, 0.3, 0.3))
            for shape in body.collision:
                shape.add_simulator_property(
                    MujocoGeom(
                        contact_type=ContactCategories(2),
                        contact_affinity=ContactCategories.DEFAULT,
                    )
                )
            world.add_connection(FixedConnection(parent=world.root, child=body))
    return world


def collision_groups(stage: Usd.Stage) -> list:
    return [
        UsdPhysics.CollisionGroup(prim)
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.CollisionGroup)
    ]


def group_of(stage: Usd.Stage, body: Body) -> UsdPhysics.CollisionGroup:
    [shape] = [
        prim
        for prim in Usd.PrimRange(prim_of(stage, body))
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    [group] = [
        group
        for group in collision_groups(stage)
        if shape.GetPath()
        in group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
    ]
    return group


def test_shapes_of_a_category_that_does_not_accept_itself_do_not_collide(tmp_path):
    world = _furniture_world()
    stage = built(world, tmp_path)
    cabinet = group_of(stage, world.get_body_by_name("cabinet"))
    drawer = group_of(stage, world.get_body_by_name("drawer"))
    assert cabinet.GetPath() == drawer.GetPath()
    assert cabinet.GetPath() in cabinet.GetFilteredGroupsRel().GetTargets()


def test_shapes_of_their_own_category_still_collide_with_default_shapes(tmp_path):
    world = _furniture_world()
    stage = built(world, tmp_path)
    cabinet = group_of(stage, world.get_body_by_name("cabinet"))
    free_box = group_of(stage, world.get_body_by_name("free.box"))
    assert free_box.GetPath() not in cabinet.GetFilteredGroupsRel().GetTargets()
    assert cabinet.GetPath() not in free_box.GetFilteredGroupsRel().GetTargets()


def test_a_world_of_default_shapes_needs_no_collision_groups(tmp_path):
    stage = built(_hinge_world().world, tmp_path)
    assert collision_groups(stage) == []


# %% bodies with nothing to weigh


def test_a_moving_body_without_shapes_or_inertial_gets_the_minimum_mass(tmp_path):
    hinge_world = _hinge_world()
    world = hinge_world.world
    with world.modify_world():
        frame = Body(name=PrefixedName("camera_frame"))
        world.add_connection(
            FixedConnection(parent=hinge_world.hinge.child, child=frame)
        )
    stage = built(world, tmp_path)
    mass = UsdPhysics.MassAPI(prim_of(stage, frame))
    builder = UsdBuilder()
    assert mass.GetMassAttr().Get() == pytest.approx(builder.minimum_mass)
    assert list(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        [builder.minimum_inertia] * 3
    )


def test_a_moving_body_with_shapes_keeps_the_mass_physx_derives_from_them(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    assert not prim_of(stage, hinge_world.hinge.child).HasAPI(UsdPhysics.MassAPI)


# %% weight the simulation carries


def test_a_body_whose_weight_is_carried_accelerates_up_by_that_part_of_gravity(
    tmp_path,
):
    hinge_world = _hinge_world()
    compensation = GravityCompensation(fraction=0.5)
    hinge_world.hinge.child.add_simulator_property(compensation)
    stage = built(hinge_world.world, tmp_path)
    prim = prim_of(stage, hinge_world.hinge.child)
    assert UsdPhysxSchema.FORCE.value in authored_api_schemas(prim)
    assert tuple(prim.GetAttribute(UsdPhysxAttribute.FORCE).Get()) == pytest.approx(
        (0.0, 0.0, compensation.fraction * UsdBuilder().gravity)
    )
    assert (
        prim.GetAttribute(UsdPhysxAttribute.FORCE_MODE).Get()
        == UsdPhysxForceMode.ACCELERATION
    )
    assert prim.GetAttribute(UsdPhysxAttribute.FORCE_IN_WORLD_FRAME).Get()


def test_a_body_carrying_its_own_weight_gets_no_force(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    prim = prim_of(stage, hinge_world.hinge.child)
    assert UsdPhysxSchema.FORCE.value not in authored_api_schemas(prim)


# %% what each prim stands for in the world


def test_every_body_prim_carries_its_bodys_id(tmp_path):
    hinge_world = _hinge_world()
    stage = built(hinge_world.world, tmp_path)
    prim = prim_of(stage, hinge_world.free_box)
    assert prim.GetCustomDataByKey(UsdCustomDataKey.ID) == str(hinge_world.free_box.id)


def test_a_joint_named_after_its_degree_of_freedom_measures_it(tmp_path):
    following = _following_hinges_world()
    stage = built(following.world, tmp_path)
    prim = joint_of(stage, following.hinge).GetPrim()
    dof_id = str(following.hinge.raw_dof.id)
    assert (
        prim.GetCustomDataByKey(UsdCustomDataKey.MEASURED_DEGREE_OF_FREEDOM) == dof_id
    )
    assert (
        prim.GetCustomDataByKey(UsdCustomDataKey.COMMANDED_DEGREE_OF_FREEDOM) == dof_id
    )


def test_a_joint_following_another_measures_nothing(tmp_path):
    following = _following_hinges_world()
    stage = built(following.world, tmp_path)
    prim = joint_of(stage, following.mirrored).GetPrim()
    assert prim.GetCustomDataByKey(UsdCustomDataKey.MEASURED_DEGREE_OF_FREEDOM) is None


def test_a_drives_joint_measures_its_axis_and_is_commanded_through_its_velocity(
    tmp_path,
):
    driven = _driven_base_world()
    stage = built(driven.world, tmp_path)
    prim = drive_joint_of(stage, driven.drive.x).GetPrim()
    assert prim.GetCustomDataByKey(UsdCustomDataKey.MEASURED_DEGREE_OF_FREEDOM) == str(
        driven.drive.x.id
    )
    assert prim.GetCustomDataByKey(UsdCustomDataKey.COMMANDED_DEGREE_OF_FREEDOM) == str(
        driven.drive.x_velocity.id
    )
