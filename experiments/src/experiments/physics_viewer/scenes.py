"""
The worlds the physics viewer puts under MuJoCo physics: scenes described in MJCF, and
robots standing on the floor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import semantic_digital_twin
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.multi_sim import ContactCategories, MujocoGeom
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import JointDynamics
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.geometry import Color, Mesh
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Iterable

# %% where the scenes are described


class MjcfFile(PathEnumeration):
    """
    The MJCF files of semantic_digital_twin's resources the scenes are read from.
    """

    DIRECTORY = Path(semantic_digital_twin.__file__).parents[2] / "resources" / "mjcf"
    """
    The directory holding the files.
    """

    FLOOR = DIRECTORY / "floor.xml"
    """
    Nothing but the ground.
    """

    CUBE = DIRECTORY / "cube_on_the_ground.xml"
    """
    A cube dropped onto the ground.
    """

    CUPS = DIRECTORY / "jeroen_cups.xml"
    """
    A set of cups.
    """


class UrdfFile(PathEnumeration):
    """
    The URDF files of semantic_digital_twin's resources the scenes are read from.
    """

    DIRECTORY = Path(semantic_digital_twin.__file__).parents[2] / "resources" / "urdf"
    """
    The directory holding the files.
    """

    APARTMENT = DIRECTORY / "apartment.urdf"
    """
    An apartment with its walls, floor and furniture.
    """


# %% the colors of a URDF


@dataclass(frozen=True)
class UrdfMaterialColors:
    """
    Gives the meshes of a world the colors the URDF it was read from declares.

    The URDF parser gives a material's color to a box, a sphere or a cylinder and leaves a
    mesh white, so the meshes of a robot or an apartment all look white.
    """

    file: str
    """
    The URDF file, or the ROS path of one.
    """

    def apply(self, world: World) -> None:
        """
        Color the meshes of the bodies whose links the URDF gives a material.

        :param world: The world the URDF was read into.
        """
        parsed = URDFParser.from_file(self.file).parsed
        named = {
            material.name: material.color.rgba
            for material in parsed.materials
            if material.color
        }
        for link in parsed.links:
            body = world.get_body_by_name(link.name)
            for visual, shape in zip(link.visuals, body.visual):
                rgba = self.rgba_of(visual.material, named)
                if rgba is not None and isinstance(shape, Mesh):
                    shape.color = Color(*rgba)

    @staticmethod
    def rgba_of(material, named: dict[str, list[float]]) -> list[float] | None:
        """
        :param material: The material of a visual, if it has one.
        :param named: The colors of the URDF's own materials, by their names.
        :return: The color the material declares itself or by its name, if any.
        """
        if material is None:
            return None
        return material.color.rgba if material.color else named.get(material.name)


# %% furniture that rests


@dataclass(frozen=True)
class RestingFurniture:
    """
    Makes the furniture of a scene stay where it is in physics.

    A drawer's collision shape overlaps the cabinet it slides in, which throws every drawer
    and door open the moment the simulation starts, so the shapes of the furniture are put
    in a category of their own, which collides with the default category and not with
    itself. Its joints get the friction that keeps a door from swinging under its own weight.
    """

    category: ContactCategories = ContactCategories(2)
    """
    The category of the furniture's collision shapes.
    """

    dynamics: JointDynamics = field(
        default_factory=lambda: JointDynamics(dry_friction=1.0, damping=0.5)
    )
    """
    The friction of the furniture's joints.
    """

    def apply(self, world: World, bodies: Iterable[Body]) -> None:
        """
        Make the bodies rest.

        :param world: The world the bodies are in.
        :param bodies: The bodies of the furniture.
        """
        furniture = list(bodies)
        for body in furniture:
            for shape in body.collision:
                shape.add_simulator_property(
                    MujocoGeom(
                        contact_type=self.category,
                        contact_affinity=ContactCategories.DEFAULT,
                    )
                )
        for connection in world.connections:
            if (
                isinstance(connection, ActiveConnection1DOF)
                and connection.child in furniture
            ):
                connection.dynamics = self.dynamics


# %% how a scene's world is made


class SceneWorld(ABC):
    """
    How the world of a scene is made.
    """

    @abstractmethod
    def build(self) -> World:
        """
        :return: A new world of the scene.
        """


@dataclass(frozen=True)
class MjcfScene(SceneWorld):
    """
    A scene described in an MJCF file.
    """

    file: Path
    """
    The file describing the scene.
    """

    def build(self) -> World:
        return MJCFParser(str(self.file)).parse()


@dataclass(frozen=True)
class UrdfScene(SceneWorld):
    """
    A scene described in a URDF file.
    """

    file: Path
    """
    The file describing the scene.
    """

    def build(self) -> World:
        world = URDFParser.from_file(str(self.file)).parse()
        UrdfMaterialColors(str(self.file)).apply(world)
        RestingFurniture().apply(world, world.bodies)
        return world


@dataclass(frozen=True)
class FloorPosition:
    """
    A position on the floor of a world.
    """

    x: float = 0.0
    """
    The distance along the world's x axis, in meters.
    """

    y: float = 0.0
    """
    The distance along the world's y axis, in meters.
    """


@dataclass(frozen=True)
class RobotInScene(SceneWorld):
    """
    A robot spawned into another scene, at a position on the floor, with its arms parked.
    """

    scene: SceneWorld
    """
    The scene the robot is spawned into.
    """

    robot: type[AbstractRobot]
    """
    The robot to spawn.
    """

    position: FloorPosition = FloorPosition()
    """
    Where the robot stands.
    """

    def build(self) -> World:
        world = self.scene.build()
        robot = RobotSpecification(
            self.robot,
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                self.position.x, self.position.y, 0.0
            ),
        ).spawn(world)
        UrdfMaterialColors(self.robot.get_ros_file_path()).apply(world)
        for arm in robot.all_arms:
            arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
        return world


# %% the scenes


class PhysicsScene(Enum):
    """
    The scenes the physics viewer offers, each by the way its world is made.
    """

    FLOOR = MjcfScene(MjcfFile.FLOOR)
    """
    Nothing but the ground.
    """

    CUBE = MjcfScene(MjcfFile.CUBE)
    """
    A cube dropped onto the ground.
    """

    CUPS = MjcfScene(MjcfFile.CUPS)
    """
    A set of cups.
    """

    APARTMENT = UrdfScene(UrdfFile.APARTMENT)
    """
    An apartment.
    """

    PR2_IN_APARTMENT = RobotInScene(
        UrdfScene(UrdfFile.APARTMENT), PR2, FloorPosition(1.5, 2.5)
    )
    """
    The PR2 in the apartment, where the demos have it stand.
    """

    PR2 = RobotInScene(MjcfScene(MjcfFile.FLOOR), PR2)
    """
    The PR2 on the floor.
    """

    TRACY = RobotInScene(MjcfScene(MjcfFile.FLOOR), Tracy)
    """
    The Tracy robot on the floor.
    """

    @classmethod
    def from_command_line(cls, name: str) -> PhysicsScene:
        """
        :param name: The scene's name in lower case, as typed.
        :return: The scene.
        :raises KeyError: When no scene has the name.
        """
        return cls[name.upper()]
