from __future__ import annotations
import logging

import matplotlib.pyplot as plt

from .world import World

logger = logging.getLogger(__name__)

import enum
import itertools
import time
from dataclasses import dataclass, field
from functools import reduce
from operator import or_
from typing import List, Iterator, Optional, Dict, Self

import numpy as np
import plotly.graph_objects as go
import rustworkx as rx
from random_events.interval import reals, SimpleInterval, Bound
from random_events.product_algebra import SimpleEvent, Event
from random_events.variable import Continuous
from rtree import index
from sortedcontainers import SortedSet

from .spatial_types import Point3


class PoseOccupiedError(Exception):
    """
    Error that is raised when a pose is occupied or not in the search space of a Connectivity Graphs.
    """

    def __init__(self, point: Point3):
        """
        Construct a new pose occupied error.
        :param pose: The pose that is occupied.
        """
        super().__init__(f"The pose {point} is occupied.")
        self.point = point


class SpatialVariables(enum.Enum):
    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")


@dataclass
class BoundingBox:
    min_x: float
    """
    The minimum x-coordinate of the bounding box.
    """

    min_y: float
    """
    The minimum y-coordinate of the bounding box.
    """

    min_z: float
    """
    The minimum z-coordinate of the bounding box.
    """

    max_x: float
    """
    The maximum x-coordinate of the bounding box.
    """

    max_y: float
    """
    The maximum y-coordinate of the bounding box.
    """

    max_z: float
    """
    The maximum z-coordinate of the bounding box.
    """

    def __hash__(self):
        # The hash should be this since comparing those via hash is checking if those are the same and not just equal
        return hash((self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z))

    @property
    def x_interval(self) -> SimpleInterval:
        """
        :return: The x interval of the bounding box.
        """
        return SimpleInterval(self.min_x, self.max_x, Bound.CLOSED, Bound.CLOSED)

    @property
    def y_interval(self) -> SimpleInterval:
        """
        :return: The y interval of the bounding box.
        """
        return SimpleInterval(self.min_y, self.max_y, Bound.CLOSED, Bound.CLOSED)

    @property
    def z_interval(self) -> SimpleInterval:
        """
        :return: The z interval of the bounding box.
        """
        return SimpleInterval(self.min_z, self.max_z, Bound.CLOSED, Bound.CLOSED)

    @property
    def simple_event(self) -> SimpleEvent:
        """
        :return: The bounding box as a random event.
        """
        return SimpleEvent({SpatialVariables.x.value: self.x_interval,
                            SpatialVariables.y.value: self.y_interval,
                            SpatialVariables.z.value: self.z_interval})

    def bloat(self, x_amount: float = 0., y_amount: float = 0, z_amount: float = 0) -> BoundingBox:
        """
        Enlarges the bounding box by a given amount in all dimensions.

        :param x_amount: The amount to adjust minimum and maximum x-coordinates
        :param y_amount: The amount to adjust minimum and maximum y-coordinates
        :param z_amount: The amount to adjust minimum and maximum z-coordinates
        :return: New enlarged bounding box
        """
        return self.__class__(self.min_x - x_amount, self.min_y - y_amount, self.min_z - z_amount,
                              self.max_x + x_amount, self.max_y + y_amount, self.max_z + z_amount)

    def contains(self, point: Point3) -> bool:
        """
        Check if the bounding box contains a point.
        """
        return self.simple_event.contains((point.x, point.y, point.z))

    def as_collection(self) -> BoundingBoxCollection:
        """
        Convert the bounding box to a collection of bounding boxes.

        :return: The bounding box as a collection
        """
        return BoundingBoxCollection([self])


@dataclass
class BoundingBoxCollection:
    """
    Dataclass for storing a collection of bounding boxes.
    """

    bounding_boxes: List[BoundingBox] = field(default_factory=list)

    def __iter__(self) -> Iterator[BoundingBox]:
        return iter(self.bounding_boxes)

    @property
    def event(self) -> Event:
        """
        :return: The bounding boxes as a random event.
        """
        return Event(*[box.simple_event for box in self.bounding_boxes])

    def merge(self, other: BoundingBoxCollection) -> BoundingBoxCollection:
        """
        Merge another bounding box collection into this one.

        :param other: The other bounding box collection.
        :return: The merged bounding box collection.
        """
        return BoundingBoxCollection(self.bounding_boxes + other.bounding_boxes)

    def bloat(self, x_amount: float = 0., y_amount: float = 0, z_amount: float = 0) -> BoundingBoxCollection:
        """
        Enlarges all bounding boxes in the collection by a given amount in all dimensions.

        :param x_amount: The amount to adjust the x-coordinates
        :param y_amount: The amount to adjust the y-coordinates
        :param z_amount: The amount to adjust the z-coordinates

        :return: The enlarged bounding box collection
        """
        return BoundingBoxCollection([box.bloat(x_amount, y_amount, z_amount) for box in self.bounding_boxes])

    @classmethod
    def from_simple_event(cls, simple_event: SimpleEvent):
        """
        Create a list of bounding boxes from a simple random event.

        :param simple_event: The random event.
        :return: The list of bounding boxes.
        """
        result = []
        for x, y, z in itertools.product(simple_event[SpatialVariables.x.value].simple_sets,
                                         simple_event[SpatialVariables.y.value].simple_sets,
                                         simple_event[SpatialVariables.z.value].simple_sets):
            result.append(BoundingBox(x.lower, y.lower, z.lower, x.upper, y.upper, z.upper))
        return result

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """
        Create a list of bounding boxes from a random event.

        :param event: The random event.
        :return: The list of bounding boxes.
        """
        return cls([box for simple_event in event.simple_sets for box in cls.from_simple_event(simple_event)])


class GraphOfConvexSets:
    """
    A graph that represents the connectivity between convex sets.

    Every node in this graph represents a convex set.
    Every edge represents an adjacency between two convex sets.
    Furthermore, the adjacency is saved in and edge attribute called "intersection".
    """

    search_space: BoundingBoxCollection
    """
    The bounding box of the search space. Defaults to the entire three dimensional space.
    """

    xy_variable = SortedSet([SpatialVariables.x.value, SpatialVariables.y.value])
    """
    The x and y variables used in our events
    """

    graph: rx.PyGraph[BoundingBox]
    """
    The connectivity graph of the convex sets.
    """

    box_to_index_map: Dict[BoundingBox, int]

    def __init__(self, search_space: Optional[BoundingBoxCollection] = None):
        self.search_space = self._make_search_space(search_space)
        self.graph = rx.PyGraph(multigraph=False)
        self.box_to_index_map = {}

    def add(self, box: BoundingBox):
        self.box_to_index_map[box] = self.graph.add_node(box)

    def calculate_connectivity(self, tolerance=0.001):
        """
        Calculate the connectivity of the graph by checking for intersections between the bounding boxes of the nodes.
        This uses an R-tree for efficient spatial indexing and intersection queries.

        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        """

        def _overlap(a_min, a_max, b_min, b_max) -> bool:
            return (a_min[0] <= b_max[0] and b_min[0] <= a_max[0] and
                    a_min[1] <= b_max[1] and b_min[1] <= a_max[1] and
                    a_min[2] <= b_max[2] and b_min[2] <= a_max[2])

        def _intersection_box(a_min, a_max, b_min, b_max):
            return BoundingBox(
                max(a_min[0], b_min[0]), max(a_min[1], b_min[1]), max(a_min[2], b_min[2]),
                min(a_max[0], b_max[0]), min(a_max[1], b_max[1]), min(a_max[2], b_max[2]),
            )

        # Build a 3-D R-tree
        prop = index.Property()
        prop.dimension = 3
        rtree_idx = index.Index(properties=prop)

        node_list = list(self.graph.nodes())
        orig_mins, orig_maxs, expanded = [], [], []

        # Record every node once, insert it into the index
        for n in node_list:
            mn = (n.min_x, n.min_y, n.min_z)
            mx = (n.max_x, n.max_y, n.max_z)
            ex = (mn[0] - tolerance, mn[1] - tolerance, mn[2] - tolerance,
                  mx[0] + tolerance, mx[1] + tolerance, mx[2] + tolerance)

            orig_mins.append(mn)
            orig_maxs.append(mx)
            expanded.append(ex)
            rtree_idx.insert(len(orig_mins) - 1, ex)

        # Query & link, skip self-loops and symmetric pairs
        for i, (mn_i, mx_i, ex_i) in enumerate(zip(orig_mins, orig_maxs, expanded)):
            for j in rtree_idx.intersection(ex_i):
                if j <= i:  # symmetry → skip
                    continue
                mn_j, mx_j = orig_mins[j], orig_maxs[j]
                if not _overlap(mn_i, mx_i, mn_j, mx_j):
                    continue  # no true overlap
                box = _intersection_box(mn_i, mx_i, mn_j, mx_j)

                # TODO check how to get the correct box here
                self.graph.add_edge(self.box_to_index_map[self.graph[i]], self.box_to_index_map[self.graph[j]], box)
        self.draw()
        print

    def draw(self):
        import rustworkx.visualization
        rustworkx.visualization.mpl_draw(self.graph)
        plt.show()

    def plot_free_space(self) -> List[go.Mesh3d]:
        """
        Plot the free space of the environment in blue.
        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event(*[node.simple_event for node in self.graph.nodes()])
        return free_space.plot(color="blue")

    def plot_occupied_space(self) -> List[go.Mesh3d]:
        """
        Plot the occupied space of the environment in red.
        :return: A list of traces that can be put into a plotly figure.
        """
        free_space = Event(*[node.simple_event for node in self.graph.nodes()])
        occupied_space = ~free_space & self.search_space.event
        return occupied_space.plot(color="red")

    def node_of_point(self, point: Point3) -> Optional[BoundingBox]:
        """
        Find the node that contains a point.

        :return: The node that contains the point or None if no node contains the point.
        """
        for node in self.graph.nodes():
            if node.contains(point):
                return node
        return None

    def path_from_to(self, start: Point3, goal: Point3) -> Optional[List[Point3]]:
        """
        Calculate a connected path from a start pose to a goal pose.

        :param start: The start pose.
        :param goal: The goal pose.
        :return: The path as a sequence of points to navigate to or None if no path exists.
        """

        # get poses from params
        start_node = self.node_of_point(start)
        goal_node = self.node_of_point(goal)

        # validate if the poses are part of the graph
        if start_node is None:
            raise PoseOccupiedError(start)
        if goal_node is None:
            raise PoseOccupiedError(goal)

        if start_node == goal_node:
            return [start, goal]

        # get the shortest path (perhaps replace with a*?)
        paths = rx.all_shortest_paths(self.graph, self.box_to_index_map[start_node], self.box_to_index_map[goal_node])

        # if it is not possible to find a path
        if len(paths) == 0:
            return None

        path = paths[0]

        # build the path
        result = [start]

        for source, target in zip(path, path[1:]):

            intersection: BoundingBox = self.graph.get_edge_data(source, target)
            x_target = intersection.x_interval.center()
            y_target = intersection.y_interval.center()
            z_target = intersection.z_interval.center()
            result.append(Point3.from_xyz(x_target, y_target, z_target))

        result.append(goal)
        return result

    @classmethod
    def _make_search_space(cls, search_space: Optional[BoundingBoxCollection] = None):
        """
        Create the default search space if it is not given.
        """
        if search_space is None:
            search_space = BoundingBox(-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf).as_collection()
        return search_space

    @classmethod
    def obstacles_of_world(cls, world: World, search_space: Optional[BoundingBoxCollection] = None,
                           bloat_obstacles: float = 0., bloat_walls: float = 0.,
                           filter_links: Optional[callable] = None) -> Event:
        """
        Get all obstacles of the world besides the robot as a random event.

        :param world: The world to get the obstacles from.
        :param search_space: The search space for the connectivity graph.
        :param bloat_obstacles: The amount to bloat the obstacles.
        :param bloat_walls: The amount to bloat the walls.
        :param filter_links: A function that filters the links to consider for obstacles.
                             If None, all links are considered.
        """
        if filter_links is None:
            filter_links = lambda link: True

        def bloat_bb(bb, link):
            if any(k in link.lower() for k in {"wall", "door"}):
                # bloat only the thickness of the wall or door, not the height or width
                if bb.width > bb.depth:
                    return bb.bloat(bloat_walls, 0, 0.01)
                else:
                    return bb.bloat(0, bloat_walls, 0.01)
            else:
                return bb.bloat(bloat_obstacles, bloat_obstacles, 0.01)

        bloated_bbs = (
            # bloat the bb
            bloat_bb(bb, link)
            # for all objects that are not robots
            for obj in world.bodies if not obj.is_a_robot
            # if the link of the object is either a wall or a door, and skip if not
            for link in obj.link_name_to_id.keys() if filter_links(link)
            for bb in obj.get_link_axis_aligned_bounding_box_collection(link)
        )

        search_space = cls._make_search_space(search_space)
        return cls.obstacles_from_bounding_boxes(list(bloated_bbs), search_space.event)

    @classmethod
    def obstacles_from_bounding_boxes(cls, bounding_boxes: List[BoundingBox], search_space_event: Event,
                                      keep_z: bool = True) -> Optional[Event]:
        """
        Create a connectivity graph from a list of bounding boxes.

        :param bounding_boxes: The list of bounding boxes to create the connectivity graph from.
        :param search_space_event: The search space event to limit the connectivity graph to.
        :param keep_z: If True, the z-axis is kept in the resulting event. Default is True.

        :return: An event representing the obstacles in the search space, or None if no obstacles are found.
        """

        if not keep_z:
            search_space_event = search_space_event.marginal(cls.xy_variable)

        events = (
            bb.simple_event.as_composite_set() & search_space_event
            for bb in bounding_boxes
        )

        # skip bbs outside the search space
        events = (event for event in events if not event.is_empty())

        if not keep_z:
            events = (event.marginal(cls.xy_variable) for event in events)

        try:
            return reduce(or_, events)
        except TypeError:
            return None

    @classmethod
    def free_space_from_world(cls, world: World, tolerance=.001, search_space: Optional[BoundingBoxCollection] = None,
                              bloat_obstacles: float = 0., bloat_walls: float = 0.) -> Self:
        """
        Create a connectivity graph from the free space in the belief state of the robot.

        :param world: The belief state.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param search_space: The search space for the connectivity graph.
        :return: The connectivity graph.
        """

        # create search space for calculations
        search_space = cls._make_search_space(search_space)
        search_event = search_space.event

        # get obstacles
        obstacles = cls.obstacles_of_world(world, search_space, bloat_obstacles, bloat_walls)

        start_time = time.time_ns()
        # calculate the free space and limit it to the searching space
        free_space = ~obstacles & search_event
        logger.info(f"Free space calculated in {(time.time_ns() - start_time) / 1e6} ms")

        # create a connectivity graph from the free space and calculate the edges
        result = cls(search_space=search_space)
        [result.add(bb) for bb in BoundingBoxCollection.from_event(free_space)]

        start_time = time.time_ns()
        result.calculate_connectivity(tolerance)
        logger.info(f"Connectivity calculated in {(time.time_ns() - start_time) / 1e6} ms")

        return result

    @classmethod
    def navigation_map_from_world(cls, world: World, tolerance=.001,
                                  search_space: Optional[BoundingBoxCollection] = None,
                                  bloat_obstacles: float = 0.) -> Self:
        """
        Create a GCS from the free space in the belief state of the robot for navigation.
        The resulting GCS describes the paths for navigation, meaning that changing the z-axis position is not
        possible.
        Furthermore, it is taken into account that the robot has to fit through the entire space and not just
        through the floor level obstacles.

        :param world: The belief state.
        :param tolerance: The tolerance for the intersection when calculating the connectivity.
        :param search_space: The search space for the connectivity graph.
        :return: The connectivity graph.
        """

        # create search space for calculations
        search_space = cls._make_search_space(search_space)

        # remove the z axis
        og_search_event = search_space.event
        search_event = og_search_event.marginal(cls.xy_variable)

        bloated_bbs = (
            # bloat the bb
            bb.bloat(bloat_obstacles, bloat_obstacles, 0.)
            # for all objects that are not robots or the floor
            for obj in world.objects if not (obj.is_a_robot or obj.name == "floor")
            # if the link of the object is either a wall or a door, and skip if not
            for link in obj.link_name_to_id.keys()
            for bb in obj.get_link_axis_aligned_bounding_box_collection(link)
        )

        obstacles = cls.obstacles_from_bounding_boxes(list(bloated_bbs), og_search_event, keep_z=False)

        free_space = ~obstacles & search_event

        # create floor level
        z_event = SimpleEvent({BoundingBox.z_variable: reals()}).as_composite_set()
        z_event.fill_missing_variables(cls.xy_variable)
        free_space.fill_missing_variables(SortedSet([BoundingBox.z_variable]))
        free_space &= z_event
        free_space &= og_search_event

        # create a connectivity graph from the free space and calculate the edges
        result = cls(search_space=search_space)
        result.add_nodes_from(BoundingBox.from_event(free_space))
        result.calculate_connectivity(tolerance)

        return result
