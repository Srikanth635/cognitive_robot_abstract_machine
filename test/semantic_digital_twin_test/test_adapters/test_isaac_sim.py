"""
Tests for simulating a world under Isaac Sim's PhysX, stepped in lockstep the way a
controller steps it.

They need Isaac Sim, which starts once, headless, before its modules can be imported,
and are skipped where it is not installed.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pytest

pytest.importorskip("isaacsim")

from semantic_digital_twin.adapters.isaac.application import IsaacApplication

IsaacApplication(headless=True).start()

from semantic_digital_twin.adapters.isaac.simulation import IsaacSim
from semantic_digital_twin.adapters.isaac.exceptions import (
    DurationNotWholeStepsError,
    WorldStructureChangedUnderIsaacError,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connection_properties import (
    VelocityServoGains,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    GravityCompensation,
)

from .test_usd_builder import DrivenBaseWorld, _driven_base_world, _hinge_world

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


# %% servos


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
