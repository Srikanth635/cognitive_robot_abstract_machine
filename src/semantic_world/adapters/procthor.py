import re
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar, Generic, List

import numpy as np
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent

from ..connections import PrismaticConnection, UnitVector, FixedConnection, RevoluteConnection
from ..geometry import Box, Scale, BoundingBoxCollection
from ..prefixed_name import PrefixedName
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import TransformationMatrix
from ..utils import IDGenerator
from ..variables import SpatialVariables
from ..views import Container, Handle, Dresser, Drawer, Door
from ..world import World
from ..world_entity import Body


id_generator = IDGenerator()

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
    name: PrefixedName
    scale: Scale = field(default_factory= lambda: Scale(1., 1., 1.))
    wall_thickness: float = 0.05
    direction: Direction = Direction.X
    def create(self) -> World:

        outer_box = event_from_scale(self.scale)
        inner_scale = Scale(self.scale.x - self.wall_thickness, self.scale.y - self.wall_thickness,
                            self.scale.z - self.wall_thickness)
        inner_box = event_from_scale(inner_scale)

        if self.direction == Direction.X:
            inner_box[SpatialVariables.x.value] = closed(-inner_scale.x / 2, self.scale.x / 2)
        elif self.direction == Direction.Y:
            inner_box[SpatialVariables.y.value] = closed(-inner_scale.y / 2, self.scale.y / 2)
        elif self.direction == Direction.Z:
            inner_box[SpatialVariables.z.value] = closed(-inner_scale.z / 2, self.scale.z / 2)
        elif self.direction == Direction.NEGATIVE_X:
            inner_box[SpatialVariables.x.value] = closed(-self.scale.x / 2, inner_scale.x / 2)
        elif self.direction == Direction.NEGATIVE_Y:
            inner_box[SpatialVariables.y.value] = closed(-self.scale.y / 2, inner_scale.y / 2)
        else:
            inner_box[SpatialVariables.z.value] = closed(-self.scale.z / 2, inner_scale.z / 2)


        container = outer_box.as_composite_set() - inner_box.as_composite_set()

        bounding_box_collection = BoundingBoxCollection.from_event(container)
        collision = bounding_box_collection.as_shapes(reference_frame=self.name)
        body = Body(name=PrefixedName(self.name.name+'_blox', self.name.prefix), collision=collision, visual=collision)
        container_view = Container(body=body, name=self.name)

        world = World()
        with world.modify_world():
            world.add_body(body)
            world.add_view(container_view)
        return world

@dataclass
class HandleFactory(ViewFactory[Handle]):
    parent_name: PrefixedName
    scale: float

    def create(self) -> World:
        name = PrefixedName(f"{self.parent_name.name}_handle", self.parent_name.prefix)
        long_side_handle = Box(origin=TransformationMatrix.from_xyz_rpy(0, 0, 0, 0, 0, 0),
                               scale=Scale(self.scale / 4, self.scale, self.scale / 4), )
        short_side_handle_left = Box(
            origin=TransformationMatrix.from_xyz_rpy(-0.25 * self.scale, 0.5 * 0.75 * self.scale, 0, 0, 0, 0),
            scale=Scale(self.scale / 4, self.scale / 4, self.scale / 4), )
        short_side_handle_right = Box(
            origin=TransformationMatrix.from_xyz_rpy(-0.25 * self.scale, -0.5 * 0.75 * self.scale, 0, 0, 0, 0),
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
class DoorFactory(ViewFactory[Door]):
    name: PrefixedName
    handle_factory: HandleFactory
    handle_transform: TransformationMatrix
    scale: Scale = field(default_factory= lambda: Scale(.03, 1., 2.))

    def create(self) -> World:

        box = event_from_scale(self.scale).as_composite_set()

        bounding_box_collection = BoundingBoxCollection.from_event(box)
        collision = bounding_box_collection.as_shapes(reference_frame=self.name)
        body = Body(name=PrefixedName(f"{self.name.name}_door_{id_generator(box)}", self.name.prefix), collision=collision,
                    visual=collision)

        world = World()
        with world.modify_world():
            world.add_body(body)

        handle_world = self.handle_factory.create()
        handle_view: Handle = handle_world.views[0]

        door_to_handle = FixedConnection(world.root, handle_world.root, self.handle_transform)

        world.merge_world(handle_world, door_to_handle)

        world.add_view(handle_view)
        world.add_view(Door(handle=handle_view, body=body))

        return world


@dataclass
class DrawerFactory(ViewFactory[Drawer]):
    parent_name: PrefixedName
    handle_factory: HandleFactory
    container_factory: ContainerFactory

    def create(self) -> World:
        name = PrefixedName(f"{self.parent_name.name}_drawer", self.parent_name.prefix)
        container_world = self.container_factory.create()
        container_view: Container = container_world.views[0]

        handle_world = self.handle_factory.create()
        handle_view: Handle = handle_world.views[0]

        drawer_to_handle = FixedConnection(container_world.root, handle_world.root,
                                           TransformationMatrix.from_xyz_rpy(0, -( self.container_factory.scale.y / 2) - 0.04, 0, 0, 0,
                                                                             -np.pi / 2, ))

        container_world.merge_world(handle_world, drawer_to_handle)

        drawer_view = Drawer(name=name, container=container_view, handle=handle_view)

        container_world.add_view(drawer_view)

        return container_world



@dataclass
class DresserFactory(ViewFactory[Dresser]):
    container_factory: ContainerFactory

    drawers_factories: List[DrawerFactory] = field(default_factory=list, hash=False)
    drawer_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)
    door_factories: List[DoorFactory] = field(default_factory=list, hash=False)
    door_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)

    def create(self) -> World:
        assert len(self.drawers_factories) == len(self.drawer_transforms), "Number of drawers must match number of transforms"

        name = PrefixedName("dresser_body")
        container_world = self.container_factory.create()
        container_view: Container = container_world.views[0]

        for drawer_factory, transform in zip(self.drawers_factories, self.drawer_transforms):
            drawer_world = drawer_factory.create()

            for drawer in drawer_world.bodies:
                if "handle" not in drawer.name.name:
                    lower_limits = DerivativeMap[float]()
                    lower_limits.position = 0.
                    upper_limits = DerivativeMap[float]()
                    upper_limits.position = drawer_factory.container_factory.scale.x * 0.75

                    dof = container_world.create_degree_of_freedom(PrefixedName(f"{drawer.name.name}_{id_generator(drawer)}_connection",
                                                                      drawer.name.prefix), lower_limits, upper_limits)
                    connection = PrismaticConnection(parent=container_world.bodies[0], child=drawer,
                                                     origin_expression=transform,
                                                     multiplier=1., offset=0., axis=UnitVector.X(),
                                                     dof=dof)
                    with container_world.modify_world():
                        container_world.merge_world(drawer_world, connection)

        for door_factory, transform in zip(self.door_factories, self.door_transforms):
            door_world = door_factory.create()

            door_view: Door = door_world.get_views_by_type(Door)[0]
            door_body = door_view.body
            handle_position = door_view.handle.body.parent_connection.origin_expression.to_position().to_np()

            lower_limits = DerivativeMap[float]()
            lower_limits.position = 0.
            upper_limits = DerivativeMap[float]()
            upper_limits.position = np.pi/2

            dof = container_world.create_degree_of_freedom(PrefixedName(f"{door_body.name.name}_{id_generator(door_body)}_connection",
                                                              door_body.name.prefix), lower_limits, upper_limits)

            relative_pivot_point = np.array([handle_position[0], -handle_position[1], handle_position[2]])
            T_DP = np.eye(4)
            T_DP[:3, 3] = relative_pivot_point
            T_WP = transform.to_np() @ T_DP

            door_position = T_WP[:3, 3]

            pivot_point = TransformationMatrix.from_xyz_rpy(door_position[0], door_position[1], door_position[2], 0, 0, 0)

            connection = RevoluteConnection(parent=container_world.bodies[0], child=door_body,
                                            origin_expression=pivot_point,
                                            multiplier=1., offset=0., axis=UnitVector.Z(),
                                            dof=dof)
            with container_world.modify_world():
                container_world.merge_world(door_world, connection)

        dresser_view = Dresser(container=container_view, drawers=[v for v in container_world.views if isinstance(v, Drawer)])
        container_world.add_view(dresser_view)

        return container_world




def replace_dresser_drawer_connections(world: World):
    dresser_pattern = re.compile(r'^dresser_\d+.*$')
    drawer_pattern = re.compile(r'^.*_drawer_.*$')
    door_pattern = re.compile(r'^.*_door_.*$')

    dresser_bodies = [b for b in world.bodies if bool(dresser_pattern.fullmatch(b.name.name))]
    for dresser in dresser_bodies:
        drawer_factories = []
        drawer_transforms = []
        door_factories = []
        door_transforms = []
        for child in dresser.child_bodies:
            if bool(drawer_pattern.fullmatch(child.name.name)):
                drawer_transforms.append(child.parent_connection.origin_expression)

                handle_factory = HandleFactory(child.name, 0.1)
                container_factory = ContainerFactory(name=child.name, scale=child.bounding_box_collection.bounding_boxes[0].scale, direction=Direction.Z)
                drawer_factory = DrawerFactory(parent_name=dresser.name, handle_factory=handle_factory, container_factory=container_factory)
                drawer_factories.append(drawer_factory)
            elif bool(door_pattern.fullmatch(child.name.name)):
                door_transforms.append(child.parent_connection.origin_expression)
                handle_factory = HandleFactory(child.name, 0.1)

                door_factory = DoorFactory(name=dresser.name, scale=child.bounding_box_collection.bounding_boxes[0].scale, handle_factory=handle_factory,
                                           handle_transform=TransformationMatrix.from_xyz_rpy(0.05, 0.4, 0, 0, 0, 0))
                door_factories.append(door_factory)


        dresser_factory = DresserFactory(container_factory=ContainerFactory(name=dresser.name, scale=dresser.bounding_box_collection.bounding_boxes[0].scale, direction=Direction.NEGATIVE_Y),
                                         drawers_factories=drawer_factories, drawer_transforms=drawer_transforms, door_factories=door_factories, door_transforms=door_transforms)

        return dresser_factory
