import re
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar, Generic

from random_events.interval import closed
from random_events.product_algebra import SimpleEvent

from ..connections import PrismaticConnection, UnitVector, FixedConnection
from ..geometry import Box, Scale, BoundingBoxCollection
from ..prefixed_name import PrefixedName
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import TransformationMatrix
from ..variables import SpatialVariables
from ..views import Container, Handle
from ..world import World
from ..world_entity import Body


class Direction(IntEnum):
    X = 0
    Y = 1
    Z = 2
    NEGATIVE_X = 3
    NEGATIVE_Y = 4
    NEGATIVE_Z = 5


def event_from_scale(scale: Scale):
    return SimpleEvent({SpatialVariables.x.value: closed(-scale.x / 2, scale.x / 2),
                             SpatialVariables.y.value: closed(-scale.y / 2, scale.y / 2),
                             SpatialVariables.z.value: closed(-scale.z / 2, scale.z / 2), })


T = TypeVar('T')

@dataclass
class ViewFactory(Generic[T], ABC):
    """
    Abstract factory for the creation of worlds containing a single view of type T.
    """

    @abstractmethod
    def create(self) -> World:
        """
        Create the world containing a view of type T.
        :return: The world.
        """
        raise NotImplementedError()


@dataclass
class ContainerFactory(ViewFactory[Container]):
    scale: Scale = field(default_factory= lambda: Scale(1., 1., 1.))
    wall_thickness: float = 0.05
    direction: Direction = Direction.X
    name: str = 'container'

    def create(self) -> World:

        outer_box = event_from_scale(self.scale)
        inner_scale = Scale(self.scale.x - self.wall_thickness, self.scale.y - self.wall_thickness,
                            self.scale.z - self.wall_thickness)
        inner_box = event_from_scale(inner_scale)

        if self.direction == Direction.X:
            open_side_event = SimpleEvent({SpatialVariables.x.value: closed(-inner_scale.x / 2, self.scale.x / 2), })
        elif self.direction == Direction.Y:
            open_side_event = SimpleEvent({SpatialVariables.y.value: closed(-inner_scale.y / 2, self.scale.y / 2), })
        elif self.direction == Direction.Z:
            open_side_event = SimpleEvent({SpatialVariables.z.value: closed(-inner_scale.z / 2, self.scale.z / 2), })
        elif self.direction == Direction.NEGATIVE_X:
            open_side_event = SimpleEvent({SpatialVariables.x.value: closed(-self.scale.x / 2, inner_scale.x / 2), })
        elif self.direction == Direction.NEGATIVE_Y:
            open_side_event = SimpleEvent({SpatialVariables.x.value: closed(-self.scale.y / 2, inner_scale.y / 2), })
        else:
            open_side_event = SimpleEvent({SpatialVariables.x.value: closed(-self.scale.z / 2, inner_scale.z / 2), })

        open_side_event.fill_missing_variables(inner_box.variables)
        inner_box = inner_box.as_composite_set() | open_side_event.as_composite_set()
        container = outer_box.as_composite_set() - inner_box

        name = PrefixedName(f"{self.name}_{self.direction.name}")
        bounding_box_collection = BoundingBoxCollection.from_event(container)
        collision = bounding_box_collection.as_shapes(reference_frame=name)
        body = Body(name=name, collision=collision, visual=collision)
        container_view = Container(body=body, name=PrefixedName(f"container_{name.name}"))

        world = World()
        with world.modify_world():
            world.add_body(body)
            world.add_view(container_view)
        return world

@dataclass
class HandleFactory(ViewFactory[Handle]):

    scale: float

    def create(self) -> World:
        name = PrefixedName("handle_body")
        long_side_handle = Box(origin=TransformationMatrix.from_xyz_rpy(0, 0, 2, 0, 0, 0),
                               scale=Scale(self.scale / 4, self.scale, self.scale / 4), )
        short_side_handle_left = Box(
            origin=TransformationMatrix.from_xyz_rpy(-0.25 * self.scale, 0.5 * 0.75 * self.scale, 2, 0, 0, 0),
            scale=Scale(self.scale / 4, self.scale / 4, self.scale / 4), )
        short_side_handle_right = Box(
            origin=TransformationMatrix.from_xyz_rpy(-0.25 * self.scale, -0.5 * 0.75 * self.scale, 2, 0, 0, 0),
            scale=Scale(self.scale / 4, self.scale / 4, self.scale / 4), )
        collision = [long_side_handle, short_side_handle_left, short_side_handle_right]
        handle = Body(name=name, collision=collision, visual=collision)
        handle_view = Handle(handle)

        world = World()
        with world.modify_world():
            world.add_body(handle)
            world.add_view(handle_view)
        return world

@dataclass
class DresserFactory(ViewFactory[...]):
    number_of_drawers: int = 2



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
                    handle_world = HandleFactory(0.1).create()
                    handle = world.bodies[0]
                    world.merge_world(handle_world)
                    drawer_to_handle = FixedConnection(drawer, handle,
                                                       TransformationMatrix.from_xyz_rpy(0.75, 0, 0, 0, 0, 0, ))
                    world.add_connection(drawer_to_handle)
    return world
