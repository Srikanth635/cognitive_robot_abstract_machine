"""
Tests for simulating a world under Isaac Sim's PhysX, stepped in lockstep the way a
controller steps it.

They need Isaac Sim, which starts once, headless, before its modules can be imported,
and are skipped where it is not installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

import pytest

pytest.importorskip("isaacsim")

from semantic_digital_twin.adapters.isaac.application import IsaacApplication

IsaacApplication(headless=True).start()

import omni.usd
from isaacsim.core.experimental.prims import Articulation
from pxr import UsdPhysics

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.adapters.usd.builder import UsdBuilder
from semantic_digital_twin.adapters.isaac.exceptions import (
    DurationNotWholeStepsError,
    WorldStructureChangedUnderIsaacError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    ServoGains,
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
    PositionServo,
)

from .test_usd_builder import (
    DrivenBaseWorld,
    _box,
    _driven_base_world,
    _hinge_world,
    _inset_door_world,
)

# %% time


def test_a_controller_tick_advances_the_physics_by_exactly_its_duration():
    simulation = started(_hinge_world().world)
    try:
        simulation.step_simulation(timedelta(milliseconds=10))
        advanced = simulation.simulator.current_simulation_time
    finally:
        simulation.stop_simulation()
    assert advanced == pytest.approx(0.01)


def test_advancing_by_part_of_a_physics_step_is_refused():
    simulation = started(_hinge_world().world)
    try:
        with pytest.raises(DurationNotWholeStepsError):
            simulation.step_simulation(
                timedelta(seconds=simulation.simulator.step_size * 1.5)
            )
    finally:
        simulation.stop_simulation()


# %% articulations


def test_an_articulation_moves_only_through_its_joints():
    simulation = started(_hinge_world().world)
    try:
        articulations = [
            Articulation(str(prim.GetPath()))
            for prim in omni.usd.get_context().get_stage().Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        shapes = [articulation.mass_matrix_shape for articulation in articulations]
        joints = [articulation.num_dofs for articulation in articulations]
    finally:
        simulation.stop_simulation()
    assert shapes == [(count, count) for count in joints]


# %% servos


def _arm_resting_on_its_mount() -> World:
    """
    :return: A world with an arm on a servoed hinge on top of the static block it is
        mounted on, sunk a little into it.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        mount = Body(name=PrefixedName("mount"))
        mount.collision = _box(mount, Scale(0.4, 0.4, 0.1))
        world.add_connection(FixedConnection(parent=root, child=mount))
        arm = Body(name=PrefixedName("arm"))
        arm.collision = _box(arm, Scale(0.3, 0.04, 0.04))
        dof = DegreeOfFreedom(
            name=PrefixedName("hinge"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-1.0, upper_position=1.0
            ),
        )
        world.add_degree_of_freedom(dof)
        world.add_connection(
            RevoluteConnection(
                name=PrefixedName("hinge"),
                parent=mount,
                child=arm,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.05, reference_frame=mount
                ),
                axis=Vector3.Z(reference_frame=arm),
                raw_dof=dof,
            )
        )
        servo = PositionServo(
            name=PrefixedName("hinge_servo"),
            gains=ServoGains(stiffness=50.0, damping=5.0, torque_limit=2.0),
        )
        servo.add_dof(dof)
        world.add_actuator(servo)
    return world


def test_an_arm_resting_on_its_mount_turns_to_its_set_point():
    world = _arm_resting_on_its_mount()
    [servo] = world.actuators
    [dof] = servo.dofs
    set_point = 0.8
    simulation = started(world)
    try:
        world.state[dof.id].position = set_point
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=2))
        reached = simulation.servo_positions([servo.name.name])[servo.name.name]
    finally:
        simulation.stop_simulation()
    assert reached == pytest.approx(set_point, abs=0.02)


def started(world) -> IsaacSim:
    """
    :return: The world's simulation, started stepped.
    """
    simulation = IsaacSim(world=world)
    simulation.start_stepped_simulation()
    return simulation


def test_a_servoed_hinge_holds_where_the_world_has_it():
    hinge_world = _hinge_world(hinge_position=0.3)
    simulation = started(hinge_world.world)
    try:
        simulation.step_simulation(timedelta(seconds=1))
        positions = simulation.servo_positions(["hinge_servo"])
    finally:
        simulation.stop_simulation()
    assert positions["hinge_servo"] == pytest.approx(0.3, abs=0.02)


def test_a_new_set_point_drives_the_hinge_there_through_the_physics():
    hinge_world = _hinge_world()
    world = hinge_world.world
    simulation = started(world)
    try:
        world.state[hinge_world.hinge.raw_dof.id].position = 0.6
        world.notify_state_change()
        simulation.step_simulation(timedelta(milliseconds=50))
        on_the_way = simulation.servo_positions(["hinge_servo"])["hinge_servo"]
        simulation.step_simulation(timedelta(seconds=2))
        arrived = simulation.servo_positions(["hinge_servo"])["hinge_servo"]
    finally:
        simulation.stop_simulation()
    assert 0.0 < on_the_way < 0.6
    assert arrived == pytest.approx(0.6, abs=0.02)
    assert world.state[hinge_world.hinge.raw_dof.id].position == 0.6


def test_the_simulation_starts_where_the_world_has_its_joints():
    driven = _driven_base_world()
    world = driven.world
    drive = driven.drive
    axes = (drive.x, drive.y, drive.yaw)
    before = [world.state[dof.id].position for dof in axes]
    simulation = started(world)
    try:
        simulation.step_simulation(timedelta(milliseconds=100))
    finally:
        simulation.stop_simulation()
    assert [world.state[dof.id].position for dof in axes] == pytest.approx(
        before, abs=0.005
    )


# %% free bodies


def test_a_free_box_falls_onto_the_base_and_the_world_follows_it():
    hinge_world = _hinge_world()
    world = hinge_world.world
    simulation = started(world)
    try:
        simulation.step_simulation(timedelta(seconds=2))
    finally:
        simulation.stop_simulation()
    [base_shape] = hinge_world.base.collision.shapes
    [box_shape] = hinge_world.free_box.collision.shapes
    resting_height = base_shape.scale.z / 2 + box_shape.scale.z / 2
    assert float(hinge_world.free_box.global_pose.z) == pytest.approx(
        resting_height, abs=0.01
    )


def test_a_free_box_whose_weight_is_carried_stays_where_it_is():
    hinge_world = _hinge_world()
    world = hinge_world.world
    hinge_world.free_box.add_simulator_property(GravityCompensation(fraction=1.0))
    height = float(hinge_world.free_box.global_pose.z)
    simulation = started(world)
    try:
        simulation.step_simulation(timedelta(seconds=1))
    finally:
        simulation.stop_simulation()
    assert float(hinge_world.free_box.global_pose.z) == pytest.approx(height, abs=0.001)


# %% joint friction


@dataclass
class FlapWorld:
    """
    A world with a flap of 1 kg held level on a horizontal hinge, its weight turning it
    down.
    """

    world: World
    """
    The world itself.
    """

    hinge: RevoluteConnection
    """
    The hinge the flap turns on.
    """


def _flap_world(dry_friction_per_weight_torque: float) -> FlapWorld:
    """
    :param dry_friction_per_weight_torque: How many times the torque of the flap's
        weight its hinge's dry friction holds.
    """
    world = World()
    lever = 0.1
    weight_torque = Inertial().mass * UsdBuilder().gravity * lever
    with world.modify_world():
        root = Body(name=PrefixedName("world"))
        world.add_body(root)
        flap = Body(name=PrefixedName("flap"))
        flap.inertial = Inertial(
            center_of_mass=Point3(lever, 0.0, 0.0),
            inertia=InertiaTensor.from_values(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
        )
        dof = DegreeOfFreedom(
            name=PrefixedName("flap_hinge"),
            limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                lower_position=-math.pi, upper_position=math.pi
            ),
        )
        world.add_degree_of_freedom(dof)
        hinge = RevoluteConnection(
            name=PrefixedName("flap_hinge"),
            parent=root,
            child=flap,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=1.0, reference_frame=root
            ),
            axis=Vector3.Y(reference_frame=flap),
            raw_dof=dof,
            dynamics=JointDynamics(
                dry_friction=dry_friction_per_weight_torque * weight_torque
            ),
        )
        world.add_connection(hinge)
    return FlapWorld(world=world, hinge=hinge)


def test_a_hinges_friction_holds_a_flap_whose_weight_turns_it_less():
    flap = _flap_world(dry_friction_per_weight_torque=2.0)
    simulation = started(flap.world)
    try:
        simulation.step_simulation(timedelta(seconds=1))
    finally:
        simulation.stop_simulation()
    assert flap.hinge.position == pytest.approx(0.0, abs=0.01)


def test_a_flap_whose_weight_turns_it_more_than_its_hinges_friction_falls():
    flap = _flap_world(dry_friction_per_weight_torque=0.5)
    simulation = started(flap.world)
    try:
        simulation.step_simulation(timedelta(seconds=1))
    finally:
        simulation.stop_simulation()
    assert abs(flap.hinge.position) > 0.3


# %% a door that comes out of its cabinet as it opens


def test_an_inset_door_on_two_joints_stays_shut_when_nothing_pushes_it():
    inset_door = _inset_door_world()
    simulation = started(inset_door.world)
    try:
        simulation.step_simulation(timedelta(seconds=2))
    finally:
        simulation.stop_simulation()
    assert inset_door.hinge.position == pytest.approx(0.0, abs=0.01)


def test_an_inset_doors_frame_slides_out_by_its_multiplier_per_radian_the_door_turns():
    inset_door = _inset_door_world()
    world = inset_door.world
    simulation = started(world)
    try:
        world.state[inset_door.hinge.raw_dof.id].position = 0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=0.5))
        [frame_joint_path] = [
            path
            for path in simulation.simulator.joint_paths
            if path.endswith(inset_door.frame_joint.name.name)
        ]
        slid = simulation.simulator.joint_positions([frame_joint_path])[
            frame_joint_path
        ]
    finally:
        simulation.stop_simulation()
    assert inset_door.hinge.position > 0.4
    assert slid == pytest.approx(
        inset_door.frame_joint.multiplier * inset_door.hinge.position, abs=1e-5
    )


# %% a wheeled drive


def test_a_drive_commanded_forward_moves_its_base_along_its_heading_at_that_speed():
    driven = _driven_base_world()
    world = driven.world
    drive = driven.drive
    speed = 0.2
    simulation = started(world)
    try:
        world.state[drive.x_velocity.id].velocity = speed
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=1.5))
        before = [world.state[dof.id].position for dof in (drive.x, drive.y)]
        simulation.step_simulation(timedelta(seconds=0.5))
        after = [world.state[dof.id].position for dof in (drive.x, drive.y)]
    finally:
        simulation.stop_simulation()
    yaw = world.state[drive.yaw.id].position
    velocity = [(end - begin) / 0.5 for begin, end in zip(before, after)]
    assert velocity == pytest.approx(
        [speed * math.cos(yaw), speed * math.sin(yaw)], abs=0.01
    )


def _driven_base_as_heavy_as_a_full_size_robot() -> DrivenBaseWorld:
    """
    :return: The driven base, weighing as much as a full-size mobile robot and driven as
        stiffly as one.
    """
    driven = _driven_base_world()
    driven.drive.child.inertial = Inertial(
        mass=270.0, inertia=InertiaTensor.from_values(20.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    )
    for servo in driven.world.actuators:
        [degree_of_freedom] = servo.dofs
        servo.gains = (
            VelocityServoGains(damping=5000.0, torque_limit=5000.0)
            if degree_of_freedom == driven.drive.yaw
            else VelocityServoGains(damping=20000.0, torque_limit=10000.0)
        )
    return driven


def test_a_base_as_heavy_as_a_full_size_robot_turns_at_the_commanded_rate():
    driven = _driven_base_as_heavy_as_a_full_size_robot()
    world = driven.world
    yaw = driven.drive.yaw
    rate = 0.3
    simulation = started(world)
    try:
        world.state[yaw.id].velocity = rate
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=0.5))
        before = world.state[yaw.id].position
        simulation.step_simulation(timedelta(seconds=0.5))
        after = world.state[yaw.id].position
    finally:
        simulation.stop_simulation()
    assert (after - before) / 0.5 == pytest.approx(rate, abs=0.01)


def test_the_world_moving_a_drives_odometry_leaves_the_base_where_it_is():
    driven = _driven_base_world()
    world = driven.world
    x = driven.drive.x
    simulation = started(world)
    try:
        measured = world.state[x.id].position
        world.state[x.id].position = measured + 1.0
        world.notify_state_change()
        simulation.step_simulation(timedelta(milliseconds=100))
    finally:
        simulation.stop_simulation()
    assert world.state[x.id].position == pytest.approx(measured, abs=0.005)


# %% the world's structure


def test_adding_a_body_while_isaac_simulates_the_world_is_refused():
    hinge_world = _hinge_world()
    world = hinge_world.world
    simulation = started(world)
    try:
        with pytest.raises(WorldStructureChangedUnderIsaacError):
            with world.modify_world():
                world.add_connection(
                    FixedConnection(
                        parent=hinge_world.base, child=Body(name=PrefixedName("late"))
                    )
                )
    finally:
        simulation.stop_simulation()
