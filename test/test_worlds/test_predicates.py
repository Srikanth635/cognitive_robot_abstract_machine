import threading
import time
import unittest

import rclpy

from semantic_world.adapters.viz_marker import VizMarkerPublisher
from semantic_world.connections import Connection6DoF
from semantic_world.geometry import Box, Scale, Color
from semantic_world.predicates.predicates import contact
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types.spatial_types import TransformationMatrix
from semantic_world.world import World
from semantic_world.world_entity import Body


class PredicateTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("test_node")

        thread = threading.Thread(target=rclpy.spin, args=(cls.node,), daemon=True)
        thread.start()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()
        cls.node.destroy_node()

    def test_in_contact(self):
        w = World()

        b1 = Body(name=PrefixedName("b1"))
        collision1 = Box(
            scale=Scale(1.0, 1.0, 1.0),
            origin=TransformationMatrix.from_xyz_rpy(
                0,
                0,
                0.0,
                0,
                0,
                0,
                reference_frame=b1,
            ),
            color=Color(1.0, 0.0, 0.0),
        )
        b1.collision = [collision1]

        b2 = Body(name=PrefixedName("b2"))
        collision2 = Box(
            scale=Scale(1.0, 1.0, 1.0),
            origin=TransformationMatrix.from_xyz_rpy(
                0.9, 0, 0.0, 0, 0, 0, reference_frame=b2
            ),
            color=Color(0.0, 1.0, 0.0),
        )
        b2.collision = [collision2]

        b3 = Body(name=PrefixedName("b3"))
        collision3 = Box(
            scale=Scale(1.0, 1.0, 1.0),
            origin=TransformationMatrix.from_xyz_rpy(
                1.8, 0, 0.0, 0, 0, 0, reference_frame=b3
            ),
            color=Color(0.0, 0.0, 1.0),
        )
        b3.collision = [collision3]

        with w.modify_world():
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)
            w.add_kinematic_structure_entity(b3)
            w.add_connection(Connection6DoF(b1, b2, _world=w))
            w.add_connection(Connection6DoF(b2, b3, _world=w))
        viz = VizMarkerPublisher(world=w, node=self.node)
        time.sleep(1.0)
        assert contact(b1, b2)
        assert not contact(b1, b3)
        assert contact(b2, b3)
        viz._stop_publishing()


if __name__ == "__main__":
    unittest.main()
