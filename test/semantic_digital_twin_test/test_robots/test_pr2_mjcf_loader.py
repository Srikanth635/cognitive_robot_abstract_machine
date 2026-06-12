from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.robots.loaders.pr2_mjcf import (
    OMNIDRIVE_CONNECTION_NAME,
    PLANAR_BASE_BODY_NAMES,
    PLANAR_BASE_CONNECTION_NAMES,
    load_pr2_mjcf,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.connections import OmniDrive


PR2_MJCF = (
    Path(__file__).resolve().parents[3]
    / "semantic_digital_twin"
    / "resources"
    / "mujoco_resources"
    / "robots"
    / "pr2"
    / "pr2.xml"
)


def test_load_pr2_mjcf_creates_semantic_omnidrive():
    world = load_pr2_mjcf(PR2_MJCF, x=1.6, y=1.875, yaw=0.25)
    world.validate()

    drive = world.get_connection_by_name(OMNIDRIVE_CONNECTION_NAME)
    assert isinstance(drive, OmniDrive)
    assert drive.parent is world.root
    assert drive.child is world.get_body_by_name("base_footprint")
    assert np.isclose(world.state[drive.x.id].position, 1.6)
    assert np.isclose(world.state[drive.y.id].position, 1.875)
    assert np.isclose(world.state[drive.yaw.id].position, 0.25)

    for name in PLANAR_BASE_CONNECTION_NAMES:
        with pytest.raises(WorldEntityNotFoundError):
            world.get_connection_by_name(name)
    for name in PLANAR_BASE_BODY_NAMES:
        with pytest.raises(WorldEntityNotFoundError):
            world.get_body_by_name(name)

    assert all(
        dof.name.name not in PLANAR_BASE_CONNECTION_NAMES
        for actuator in world.actuators
        for dof in actuator.dofs
    )

    with world.modify_world():
        pr2 = PR2._init_empty_robot(world)
        pr2._setup_semantic_annotations()
        pr2._setup_hardware_interfaces()
        world.add_semantic_annotation(pr2)
    assert pr2.drive is drive
    assert pr2.drive.has_hardware_interface

    for dof in world.degrees_of_freedom:
        for limits in (dof.limits.lower, dof.limits.upper):
            for derivative in ("position", "velocity", "acceleration", "jerk"):
                value = getattr(limits, derivative)
                assert value is None or type(value) is float
