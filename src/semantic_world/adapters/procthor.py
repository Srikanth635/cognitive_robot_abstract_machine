import re
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar, Generic, List

import numpy as np
from numpy import ndarray
from pycram.world_reasoning import supporting
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, Event

from ..connections import PrismaticConnection, UnitVector, FixedConnection, RevoluteConnection
from ..geometry import Box, Scale, BoundingBoxCollection
from ..prefixed_name import PrefixedName
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import TransformationMatrix
from ..utils import IDGenerator
from ..variables import SpatialVariables
from ..views import Container, Handle, Dresser, Drawer, Door, Shelf, SupportingSurface
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
class ShelfFactory(ViewFactory[Shelf]):
    name: PrefixedName
    scale: Scale = field(default_factory=lambda: Scale(1., 1., 1.))
    supporting_surfaces_factories: List[ViewFactory[SupportingSurface]] = field(default_factory=list, hash=False)
    supporting_surfaces_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)
    drawer_factories: List[ViewFactory[Drawer]] = field(default_factory=list, hash=False)
    drawer_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)
    door_factories: List[ViewFactory[Door]] = field(default_factory=list, hash=False)
    door_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)
    #
    # def create(self) -> World:
    #
    #     container = event_from_scale(self.scale).as_composite_set()
    #
    #     for surface_factory, transform in zip(self.supporting_surfaces_factories, self.supporting_surfaces_transforms):
    #         surface_world = surface_factory.create()
    #         surface_view: SupportingSurface = surface_world.get_views_by_type(SupportingSurface)[0]
    #         surface_region_bb_collection = surface_view.region.as_bounding_box_collection()
    #         container = container - surface_region_bb_collection.event
    #
    #         bounding_box_collection = BoundingBoxCollection.from_event(container)
    #         collision = bounding_box_collection.as_shapes(reference_frame=self.name)
    #         body = Body(name=surface_body.name, collision=collision, visual=collision)
    #
    #         connection = FixedConnection(parent=surface_world.root, child=body, origin_expression=transform)




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
        body = Body(name=self.name, collision=collision, visual=collision)
        container_view = Container(body=body, name=self.name)

        world = World()
        with world.modify_world():
            world.add_body(body)
            world.add_view(container_view)
        return world

@dataclass
class HandleFactory(ViewFactory[Handle]):
    name: PrefixedName
    width: float

    def create(self) -> World:

        # I think this construction is wrong since i fixed the mesh orientation, but it doesnt visually
        # break, just results in some unintuitive rotations. fix when there is time
        long_side_handle = Box(origin=TransformationMatrix.from_xyz_rpy(0, 0, 0, 0, 0, 0),
                               scale=Scale(self.width / 4, self.width, self.width / 4), )
        short_side_handle_left = Box(
            origin=TransformationMatrix.from_xyz_rpy(-0.25 * self.width, 0.5 * 0.75 * self.width, 0, 0, 0, 0),
            scale=Scale(self.width / 4, self.width / 4, self.width / 4), )
        short_side_handle_right = Box(
            origin=TransformationMatrix.from_xyz_rpy(-0.25 * self.width, -0.5 * 0.75 * self.width, 0, 0, 0, 0),
            scale=Scale(self.width / 4, self.width / 4, self.width / 4), )
        collision = [long_side_handle, short_side_handle_left, short_side_handle_right]
        handle = Body(name=self.name, collision=collision, visual=collision)
        handle_view = Handle(name=self.name, body=handle)

        world = World()
        with world.modify_world():
            world.add_body(handle)
            world.add_view(handle_view)
        return world

@dataclass
class DoorFactory(ViewFactory[Door]):
    name: PrefixedName
    handle_factory: HandleFactory
    handle_direction: Direction
    scale: Scale = field(default_factory= lambda: Scale(.03, 1., 2.))

    def create(self) -> World:

        x_interval = closed(-self.scale.x / 2, self.scale.x / 2)
        y_interval = closed(-self.scale.y / 2, self.scale.y / 2)
        z_interval = closed(-self.scale.z / 2, self.scale.z / 2)

        if self.handle_direction == Direction.X:
            x_interval = closed(0, self.scale.x)
            handle_transform = TransformationMatrix.from_xyz_rpy(self.scale.x - 0.1, 0.05, 0, 0, 0, np.pi / 2)
        elif self.handle_direction == Direction.NEGATIVE_X:
            x_interval = closed(-self.scale.x, 0)
            handle_transform = TransformationMatrix.from_xyz_rpy(-(self.scale.x - 0.1), 0.05, 0, 0, 0, np.pi / 2)
        elif self.handle_direction == Direction.Y:
            y_interval = closed(0, self.scale.y)
            handle_transform = TransformationMatrix.from_xyz_rpy(0.05, (self.scale.y - 0.1), 0, 0, 0, 0)
        elif self.handle_direction == Direction.NEGATIVE_Y:
            y_interval = closed(-self.scale.y, 0)
            handle_transform = TransformationMatrix.from_xyz_rpy(0.05, -(self.scale.y - 0.1), 0, 0, 0, 0)
        else:
            raise NotImplementedError(f"Handle direction Z and NEGATIVE_Z are not implemented yet")

        box = SimpleEvent({SpatialVariables.x.value: x_interval,
                             SpatialVariables.y.value: y_interval,
                             SpatialVariables.z.value: z_interval, }).as_composite_set()

        bounding_box_collection = BoundingBoxCollection.from_event(box)
        collision = bounding_box_collection.as_shapes(reference_frame=self.name)
        body = Body(name=self.name, collision=collision, visual=collision)

        world = World()
        with world.modify_world():
            world.add_body(body)

        handle_world = self.handle_factory.create()
        handle_view: Handle = handle_world.get_views_by_type(Handle)[0]

        door_to_handle = FixedConnection(world.root, handle_world.root, handle_transform)

        world.merge_world(handle_world, door_to_handle)

        world.add_view(handle_view)
        world.add_view(Door(name=self.name, handle=handle_view, body=body))

        return world


@dataclass
class DrawerFactory(ViewFactory[Drawer]):
    name: PrefixedName
    handle_factory: HandleFactory
    container_factory: ContainerFactory

    def create(self) -> World:
        container_world = self.container_factory.create()
        container_view: Container = container_world.get_views_by_type(Container)[0]

        handle_world = self.handle_factory.create()
        handle_view: Handle = handle_world.get_views_by_type(Handle)[0]

        drawer_to_handle = FixedConnection(container_world.root, handle_world.root,
                                           TransformationMatrix.from_xyz_rpy(( self.container_factory.scale.x / 2) + 0.03, 0, 0, 0, 0, 0))

        container_world.merge_world(handle_world, drawer_to_handle)

        drawer_view = Drawer(name=self.name, container=container_view, handle=handle_view)

        container_world.add_view(drawer_view)

        return container_world



@dataclass
class DresserFactory(ViewFactory[Dresser]):
    name: PrefixedName
    container_factory: ContainerFactory
    drawers_factories: List[DrawerFactory] = field(default_factory=list, hash=False)
    drawer_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)
    door_factories: List[DoorFactory] = field(default_factory=list, hash=False)
    door_transforms: List[TransformationMatrix] = field(default_factory=list, hash=False)

    def create(self) -> World:
        assert len(self.drawers_factories) == len(self.drawer_transforms), "Number of drawers must match number of transforms"

        container_world = self.container_factory.create()
        container_view: Container = container_world.get_views_by_type(Container)[0]

        for drawer_factory, transform in zip(self.drawers_factories, self.drawer_transforms):
            drawer_world = drawer_factory.create()

            drawer_view: Drawer = drawer_world.get_views_by_type(Drawer)[0]
            drawer_body = drawer_view.container.body

            lower_limits = DerivativeMap[float]()
            lower_limits.position = 0.
            upper_limits = DerivativeMap[float]()
            upper_limits.position = drawer_factory.container_factory.scale.x * 0.75

            dof = container_world.create_degree_of_freedom(PrefixedName(f"{drawer_body.name.name}_connection",
                                                              drawer_body.name.prefix), lower_limits, upper_limits)
            connection = PrismaticConnection(parent=container_world.bodies[0], child=drawer_body,
                                             origin_expression=transform,
                                             multiplier=1., offset=0., axis=UnitVector.X(),
                                             dof=dof)
            with container_world.modify_world():
                container_world.merge_world(drawer_world, connection)

        for door_factory, transform in zip(self.door_factories, self.door_transforms):
            door_world = door_factory.create()

            door_view: Door = door_world.get_views_by_type(Door)[0]
            door_body = door_view.body


            handle_position: ndarray[float] = door_view.handle.body.parent_connection.origin_expression.to_position().to_np()

            lower_limits = DerivativeMap[float]()
            upper_limits = DerivativeMap[float]()
            if door_factory.handle_direction in {Direction.NEGATIVE_X, Direction.NEGATIVE_Y}:
                lower_limits.position = 0.
                upper_limits.position = np.pi/2
            else:
                lower_limits.position = -np.pi/2
                upper_limits.position = 0.

            dof = container_world.create_degree_of_freedom(PrefixedName(f"{door_body.name.name}_connection",
                                                              door_body.name.prefix), lower_limits, upper_limits)

            offset = -np.sign(handle_position[1]) * (door_factory.scale.y / 2)
            door_position = transform.to_np()[:3, 3] + np.array([0, offset, 0])

            pivot_point = TransformationMatrix.from_xyz_rpy(door_position[0], door_position[1], door_position[2], 0, 0, 0)

            connection = RevoluteConnection(parent=container_world.bodies[0], child=door_body,
                                            origin_expression=pivot_point,
                                            multiplier=1., offset=0., axis=UnitVector.Z(),
                                            dof=dof)
            with container_world.modify_world():
                container_world.merge_world(door_world, connection)

        dresser_view = Dresser(name=self.name, container=container_view,
                               drawers=[drawer for drawer in container_world.get_views_by_type(Drawer)],
                               doors=[door for door in container_world.get_views_by_type(Door)])
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

                handle_factory = HandleFactory(name=PrefixedName(child.name.name+"_handle", child.name.prefix), width=0.1)
                container_factory = ContainerFactory(name=PrefixedName(child.name.name+"_container", child.name.prefix), scale=child.bounding_box_collection.bounding_boxes[0].scale, direction=Direction.Z)
                drawer_factory = DrawerFactory(name=child.name, handle_factory=handle_factory, container_factory=container_factory)
                drawer_factories.append(drawer_factory)
            elif bool(door_pattern.fullmatch(child.name.name)):
                door_transforms.append(child.parent_connection.origin_expression)
                handle_factory = HandleFactory(PrefixedName(child.name.name+"_handle", child.name.prefix), 0.1)

                door_factory = DoorFactory(name=child.name, scale=child.bounding_box_collection.bounding_boxes[0].scale, handle_factory=handle_factory,
                                           handle_direction=Direction.Y)
                door_factories.append(door_factory)

        dresser_container_factory = ContainerFactory(name=PrefixedName(dresser.name.name+"_container", dresser.name.prefix),
                                                     scale=dresser.bounding_box_collection.bounding_boxes[0].scale, direction=Direction.X)
        dresser_factory = DresserFactory(name=dresser.name, container_factory=dresser_container_factory, drawers_factories=drawer_factories,
                                         drawer_transforms=drawer_transforms, door_factories=door_factories, door_transforms=door_transforms)

        return dresser_factory

from random_events.interval import Bound
from random_events.interval import open, open_closed
from semantic_world.variables import SpatialVariables
from random_events.product_algebra import *

def has_supporting_surfaces():

    shelf = SimpleEvent({SpatialVariables.x.value: (0, 1),
                         SpatialVariables.y.value: (0, 0.6),
                         SpatialVariables.z.value: (0, 1.5)})
    empty_space = SimpleEvent({SpatialVariables.y.value: (0, 0.6),
                               SpatialVariables.z.value: open(0.2, 0.6) | open_closed(0.8, 1.5),
                               SpatialVariables.x.value: open(0.1, 0.45) | open(0.55, 0.9)})
    shelf = shelf.as_composite_set() - empty_space.as_composite_set()

    shelf_c = ~shelf

    supporting_surfaces = None

    for simple_event in shelf_c.simple_sets:
        for x, y, z in itertools.product(simple_event[SpatialVariables.x.value].simple_sets,
                                         simple_event[SpatialVariables.y.value].simple_sets,
                                         simple_event[SpatialVariables.z.value].simple_sets):
            z: SimpleInterval
            z = SimpleInterval(z.lower, z.upper, Bound.CLOSED, z.right)

            current_box = SimpleEvent({SpatialVariables.x.value: x,
                                       SpatialVariables.y.value: y,
                                     SpatialVariables.z.value: z}).as_composite_set()

            intersection = shelf & current_box
            if not intersection.is_empty():
                if supporting_surfaces is None:
                    supporting_surfaces = intersection
                else:
                    supporting_surfaces = supporting_surfaces | intersection