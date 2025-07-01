import os
import time
import unittest

from matplotlib import pyplot as plt
from random_events.interval import SimpleInterval
from random_events.product_algebra import SimpleEvent
import rustworkx as rx

import plotly.graph_objects as go

from semantic_world.graph_of_convex_sets import GraphOfConvexSets, BoundingBox, SpatialVariables, BoundingBoxCollection
from semantic_world.spatial_types import Point3


class GCSTestCase(unittest.TestCase):
    """
    Testcase to test the navigation around a unit box.
    """

    gcs: GraphOfConvexSets

    @classmethod
    def setUpClass(cls):
        gcs = GraphOfConvexSets()

        obstacle = BoundingBox(0, 0, 0, 1, 1, 1)

        z_lim = SimpleInterval(.45, .55)
        x_lim = SimpleInterval(-2, 3)
        y_lim = SimpleInterval(-2, 3)
        limiting_event = SimpleEvent({SpatialVariables.z.value: z_lim,
                                      SpatialVariables.x.value: x_lim,
                                      SpatialVariables.y.value: y_lim})
        obstacles = BoundingBoxCollection.from_event(
            ~obstacle.simple_event.as_composite_set() & limiting_event.as_composite_set())
        [gcs.add(bb) for bb in obstacles]
        gcs.calculate_connectivity()
        cls.gcs = gcs

    def test_reachability(self):
        start_point = Point3.from_xyz(-1, -1, 0.5)
        target_point = Point3.from_xyz(2, 2, 0.5)

        path = self.gcs.path_from_to(start_point, target_point)
        self.assertEqual(len(path), 3)

    def test_plot(self):
        free_space_plot = go.Figure(self.gcs.plot_free_space())
        self.assertIsNotNone(free_space_plot)
        occupied_space_plot = go.Figure(self.gcs.plot_occupied_space())
        self.assertIsNotNone(occupied_space_plot)


class GCSFromWorldTestCase(unittest.TestCase):
    """
    Test the application of a connectivity graph to the belief state.
    """

    def test_from_world(self):
        search_space = BoundingBox(min_x=-1, max_x=1,
                                   min_y=-1, max_y=1,
                                   min_z=0.1, max_z=1).as_collection()
        gcs = GraphOfConvexSets.free_space_from_world(self.world, search_space=search_space)
        self.assertIsNotNone(gcs)
        self.assertGreater(len(gcs.nodes), 0)
        self.assertGreater(len(gcs.edges), 0)

        start = PoseStamped.from_list([-0.9, -0.9, 0.4])
        target = PoseStamped.from_list([-0.9, 0.9, 0.9])

        path = gcs.path_from_to(start, target)

        if "ROS_VERSION" in os.environ:
            pub = TrajectoryPublisher()
            pub.visualize_trajectory(path)

        self.assertIsNotNone(path)
        self.assertGreater(len(path), 1)

        with self.assertRaises(PoseOccupiedError):
            start = PoseStamped.from_list([-10, -10, -10])
            target = PoseStamped.from_list([10, 10, 10])
            gcs.path_from_to(start, target)

    def test_navigation_map_from_world(self):
        search_space = BoundingBox(min_x=-1, max_x=1,
                                   min_y=-1, max_y=1,
                                   min_z=0.1, max_z=1)
        gcs = GraphOfConvexSets.navigation_map_from_world(self.world, search_space=BoundingBoxCollection([search_space]))
        self.assertGreater(len(gcs.nodes), 0)
        self.assertGreater(len(gcs.edges), 0)


if __name__ == '__main__':
    unittest.main()