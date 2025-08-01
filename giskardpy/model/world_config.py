from __future__ import annotations

import abc
from abc import ABC
from typing import Optional, Union

import numpy as np

from giskardpy.god_map import god_map
from giskardpy.model.utils import robot_name_from_urdf_string
from semantic_world.adapters.urdf import URDFParser
from semantic_world.connections import Has1DOFState, Connection6DoF, OmniDrive
from semantic_world.geometry import Color
from semantic_world.prefixed_name import PrefixedName
from semantic_world.robots import PR2
from semantic_world.spatial_types.derivatives import Derivatives
from semantic_world.world import World
from semantic_world.world_entity import Body


class WorldConfig(ABC):
    _world: World
    default_color = Color(0.5, 0.5, 0.5, 1)

    def __init__(self, register_on_god_map: bool = True):
        self._world = World()
        if register_on_god_map:
            god_map.world = self.world

    @property
    def world(self) -> World:
        return self._world

    def set_defaults(self):
        pass

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        """
        Implement this method to configure the initial world using it's self. methods.
        """

    @property
    def robot_group_name(self) -> str:
        return self.world.robot_name

    def get_root_link_of_group(self, group_name: str) -> PrefixedName:
        return self.world.views[group_name].root_link_name

    def set_default_color(self, color: Color) -> None:
        """
        :param r: 0-1
        :param g: 0-1
        :param b: 0-1
        :param a: 0-1
        """
        self.world.default_link_color = color

    def add_robot_urdf(self,
                       urdf: str,
                       group_name: Optional[str] = None) -> str:
        """
        Add a robot urdf to the world.
        :param urdf: robot urdf as string, not the path
        :param group_name:
        """
        if group_name is None:
            group_name = robot_name_from_urdf_string(urdf)
        urdf_parser = URDFParser(urdf)
        world_with_robot = urdf_parser.parse()
        self.world.merge_world(world_with_robot)
        return group_name


class EmptyWorld(WorldConfig):
    def setup(self):
        # self._default_limits = {
        #     Derivatives.velocity: 1,
        #     Derivatives.acceleration: np.inf,
        #     Derivatives.jerk: None
        # }
        # self.set_default_limits(self._default_limits)
        self.add_empty_link(PrefixedName('map'))


class WorldWithFixedRobot(WorldConfig):
    def __init__(self,
                 urdf: str,
                 map_name: str = 'map'):
        super().__init__()
        self.urdf = urdf
        self.map_name = PrefixedName(map_name)

    def setup(self, robot_name: Optional[str] = None) -> None:
        self.set_default_limits({Derivatives.velocity: 1,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: None})
        self.add_empty_link(self.map_name)
        self.add_robot_urdf(self.urdf, robot_name)
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_fixed_joint(parent_link=self.map_name, child_link=root_link_name)


class WorldWithOmniDriveRobot(WorldConfig):
    map_name: PrefixedName
    odom_link_name: PrefixedName

    def __init__(self,
                 urdf: str,
                 map_name: str = 'map',
                 odom_link_name: str = 'odom'):
        super().__init__()
        self.urdf = urdf
        self.map_name = PrefixedName(map_name)
        self.odom_link_name = PrefixedName(odom_link_name)

    def setup(self, robot_name: Optional[str] = None):
        map = Body(name=self.map_name)
        odom = Body(name=self.odom_link_name)
        localization = Connection6DoF(parent=map, child=odom, _world=self.world)
        self.world.add_connection(localization)

        urdf_parser = URDFParser(urdf=self.urdf)
        world_with_robot = urdf_parser.parse()

        odom = OmniDrive(parent=odom, child=world_with_robot.root,
                         translation_velocity_limits=0.2,
                         rotation_velocity_limits=0.2,
                         _world=self.world)

        self.world.merge_world(world_with_robot, odom)

        PR2.from_world(world=self.world)


class WorldWithDiffDriveRobot(WorldConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(self,
                 urdf: str,
                 map_name: str = 'map',
                 localization_joint_name: str = 'localization',
                 odom_link_name: str = 'odom',
                 drive_joint_name: str = 'brumbrum'):
        super().__init__()
        self.urdf = urdf
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.set_default_limits({Derivatives.velocity: 1,
                                 Derivatives.acceleration: np.inf,
                                 Derivatives.jerk: None})
        self.add_empty_link(PrefixedName(self.map_name))
        self.add_empty_link(PrefixedName(self.odom_link_name))
        self.add_6dof_joint(parent_link=self.map_name, child_link=self.odom_link_name,
                            joint_name=self.localization_joint_name)
        self.add_robot_urdf(urdf=self.urdf)
        root_link_name = self.get_root_link_of_group(self.robot_group_name)
        self.add_diff_drive_joint(name=self.drive_joint_name,
                                  parent_link_name=self.odom_link_name,
                                  child_link_name=root_link_name,
                                  translation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: np.inf,
                                      Derivatives.jerk: None,
                                  },
                                  rotation_limits={
                                      Derivatives.velocity: 0.2,
                                      Derivatives.acceleration: np.inf,
                                      Derivatives.jerk: None
                                  },
                                  robot_group_name=self.robot_group_name)
