from pathlib import Path
from typing import Union

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive


PLANAR_BASE_CONNECTION_NAMES = (
    "base_x_joint",
    "base_y_joint",
    "base_yaw_joint",
)
PLANAR_BASE_BODY_NAMES = (
    "pr2_planar_y",
    "pr2_planar_x",
)
OMNIDRIVE_CONNECTION_NAME = "odom_combined_T_base_footprint"


def load_pr2_mjcf(
    file_path: Union[str, Path],
    *,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
) -> World:
    """
    Load the generated PR2 MJCF as an SDT world with a semantic OmniDrive.

    The generated MJCF represents planar base motion as serial x/y slide joints
    followed by a yaw hinge. SDT, PyCRAM, and Giskard represent the same
    capability as one OmniDrive connection to ``base_footprint``.

    :param file_path: Path to the generated PR2 MJCF.
    :param x: Initial world-frame base x position.
    :param y: Initial world-frame base y position.
    :param yaw: Initial world-frame base yaw angle in radians.
    :return: Parsed PR2 world with the planar joint chain replaced by OmniDrive.
    """
    world = MJCFParser(str(file_path)).parse()
    _normalize_dof_limits(world)
    _replace_planar_base_with_omnidrive(world, x=x, y=y, yaw=yaw)
    return world


def _normalize_dof_limits(world: World) -> None:
    """Convert parser-produced numeric scalar limits to native Python floats."""
    for dof in world.degrees_of_freedom:
        for limits in (dof.limits.lower, dof.limits.upper):
            for derivative in ("position", "velocity", "acceleration", "jerk"):
                value = getattr(limits, derivative)
                if value is not None:
                    setattr(limits, derivative, float(value))


def _replace_planar_base_with_omnidrive(
    world: World,
    *,
    x: float,
    y: float,
    yaw: float,
) -> OmniDrive:
    world_root = world.root
    base_footprint = world.get_body_by_name("base_footprint")
    planar_connections = [
        world.get_connection_by_name(name)
        for name in PLANAR_BASE_CONNECTION_NAMES
    ]
    planar_dofs = {
        dof for connection in planar_connections for dof in connection.dofs
    }

    with world.modify_world():
        for actuator in list(world.actuators):
            if any(dof in planar_dofs for dof in actuator.dofs):
                world.remove_actuator(actuator)
        for connection in planar_connections:
            world.remove_connection(connection)
        for body_name in PLANAR_BASE_BODY_NAMES:
            world.remove_kinematic_structure_entity(
                world.get_body_by_name(body_name)
            )
        world.delete_orphaned_dofs()

        drive = OmniDrive.create_with_dofs(
            world=world,
            parent=world_root,
            child=base_footprint,
            name=PrefixedName(OMNIDRIVE_CONNECTION_NAME),
        )
        world.add_connection(drive)
        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            reference_frame=world_root,
        )

    return drive
