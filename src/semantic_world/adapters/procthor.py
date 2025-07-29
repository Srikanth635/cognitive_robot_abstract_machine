import re

from ..connections import PrismaticConnection, UnitVector, FixedConnection
from ..geometry import Box, Scale
from ..prefixed_name import PrefixedName
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import TransformationMatrix
from ..world import World
from ..world_entity import Body


def make_handle_body(scale):
    name = PrefixedName("handle_body")
    long_side_handle = Box(origin=TransformationMatrix.from_xyz_rpy(0, 0, 2, 0, 0, 0),
                           scale=Scale(scale/4, scale, scale/4),)
    short_side_handle_left = Box(origin=TransformationMatrix.from_xyz_rpy(-0.25 * scale, 0.5 * 0.75 * scale, 2, 0, 0, 0),
                                 scale=Scale(scale/4, scale/4, scale/4),)
    short_side_handle_right = Box(origin=TransformationMatrix.from_xyz_rpy(-0.25 * scale, -0.5 * 0.75 * scale, 2, 0, 0, 0),
                                 scale=Scale(scale/4, scale/4, scale/4),)
    collision = [long_side_handle, short_side_handle_left, short_side_handle_right]
    handle = Body(name=name, collision=collision, visual=collision)
    return handle

def make_env():
    for i in range(10):
        h = make_handle_body(0.2)


def replace_dresser_drawer_connections(world: World):
    dresser_pattern = re.compile(r'^dresser_\d+$')
    drawer_pattern = re.compile(r'^.*_drawer_.*$')

    dresser_bodies = [b for b in world.bodies if bool(dresser_pattern.fullmatch(b.name.name))]
    for dresser in dresser_bodies:
        for drawer in dresser.child_bodies:
            if bool(drawer_pattern.fullmatch(drawer.name.name)):
                lower_limits = DerivativeMap[float]()
                lower_limits.position = 0.
                upper_limits = DerivativeMap[float]()
                upper_limits.position = drawer.bounding_box_collection.bounding_boxes[0].depth * 0.75

                dof = world.create_degree_of_freedom(PrefixedName(f"{drawer.name.name}_connection",
                                                                  drawer.name.prefix), lower_limits, upper_limits)
                connection = PrismaticConnection(parent=dresser, child=drawer,
                                                 origin_expression=drawer.parent_connection.origin_expression,
                                                 multiplier=-1., offset=0., axis=UnitVector.X(),
                                                 dof=dof)

                with world.modify_world():
                    original_connection = world.get_connection(dresser, drawer)
                    world.remove_connection(original_connection)
                    world.add_connection(connection)
                    handle = make_handle_body(0.1)
                    world.add_body(handle)
                    drawer_to_handle = FixedConnection(drawer, handle, TransformationMatrix.from_xyz_rpy(0.75, 0, 0, 0, 0, 0, ))
                    world.add_connection(drawer_to_handle)
    return world
