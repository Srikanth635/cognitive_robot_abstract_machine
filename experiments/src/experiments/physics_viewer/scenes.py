"""
The worlds the physics viewer puts under MuJoCo physics: scenes described in MJCF, and
robots standing on the floor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import semantic_digital_twin
from coraplex.testing import setup_world
from krrood.patterns.path_enumeration import PathEnumeration
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World

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
        return URDFParser.from_file(str(self.file)).parse()


@dataclass(frozen=True)
class DemoWorld(SceneWorld):
    """
    The world the demos perform their plans in: the PR2 in the apartment, with a milk
    carton and a cereal box.
    """

    def build(self) -> World:
        return setup_world()


@dataclass(frozen=True)
class RobotOnTheFloor(SceneWorld):
    """
    A robot spawned onto the floor, at its own origin.
    """

    robot: type[AbstractRobot]
    """
    The robot to spawn.
    """

    def build(self) -> World:
        world = MjcfScene(MjcfFile.FLOOR).build()
        RobotSpecification(self.robot).spawn(world)
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

    PR2_IN_APARTMENT = DemoWorld()
    """
    The PR2 in the apartment with a milk carton and a cereal box, as the demos have it.
    """

    PR2 = RobotOnTheFloor(PR2)
    """
    The PR2 on the floor.
    """

    TRACY = RobotOnTheFloor(Tracy)
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
