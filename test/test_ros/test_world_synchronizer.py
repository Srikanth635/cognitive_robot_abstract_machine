import threading
import time
import unittest
from unittest import skipUnless

import sqlalchemy
from ormatic.utils import drop_database
from sqlalchemy import select
from sqlalchemy.orm import Session

from semantic_world.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_world.connections import Connection6DoF
from semantic_world.orm.ormatic_interface import Base, WorldMappingDAO
from semantic_world.prefixed_name import PrefixedName
from semantic_world.utils import rclpy_installed
from semantic_world.world import World
from semantic_world.world_entity import Body

if rclpy_installed():
    import rclpy
    from uuid import uuid4


@skipUnless(rclpy_installed(), "rclpy required")
class WorldStatePublisherTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        # cls.node = rclpy.create_node("WorldStatePublisher")
        # cls.synch_thread = threading.Thread(
        #     target=rclpy.spin, args=(cls.node,), daemon=False
        # )
        # cls.synch_thread.start()

    def setUp(self):
        # Create an isolated node per test to avoid cross-talk across tests
        self.node = rclpy.create_node(f"WorldStatePublisher_{uuid4().hex}")
        self.synch_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self.synch_thread.start()

    def tearDown(self):
        # Ensure all subscriptions/publishers are destroyed between tests
        self.node.destroy_node()
        # Give the spin thread a moment to exit
        self.synch_thread.join(timeout=1.0)


    @staticmethod
    def create_dummy_world():
        w = World()
        b1 = Body(name=PrefixedName("b1"))
        b2 = Body(name=PrefixedName("b2"))
        with w.modify_world():
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)
            w.add_connection(Connection6DoF(b1, b2, _world=w))
        return w

    def test_state_synchronization(self):
        w1 = self.create_dummy_world()
        w2 = self.create_dummy_world()

        synchronizer_1 = WorldSynchronizer(self.node, w1, subscribe=False)
        synchronizer_2 = WorldSynchronizer(self.node, w2)

        w1.state.data[0, 0] = 1.0
        w1.notify_state_change()
        time.sleep(2.0)
        assert w1.state.data[0, 0] == 1.0
        assert w1.state.data[0, 0] == w2.state.data[0, 0]

    def test_model_reload(self):
        engine = sqlalchemy.create_engine(
            "sqlite+pysqlite:///file::memory:?cache=shared",
            connect_args={"check_same_thread": False, "uri": True},
        )
        drop_database(engine)
        Base.metadata.create_all(engine)
        session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
        session1 = session_maker()
        session2 = session_maker()

        w1 = self.create_dummy_world()
        w2 = World()

        synchronizer_1 = WorldSynchronizer(
            self.node, w1, subscribe=False, session=session1
        )
        synchronizer_2 = WorldSynchronizer(self.node, w2, session=session2)

        synchronizer_1.publish_reload_model()
        time.sleep(0.1)
        self.assertEqual(len(w2.kinematic_structure_entities), 2)

        query = session1.scalars(select(WorldMappingDAO)).all()
        assert len(query) == 1
        assert w2.get_kinematic_structure_entity_by_name("b2")

    def test_model_synchronization_body_only(self):

        w1 = World(name="w1")
        w2 = World(name="w2")

        synchronizer_1 = WorldSynchronizer(
            self.node, w1, subscribe=False
        )
        synchronizer_2 = WorldSynchronizer(self.node, w2)

        with w1.modify_world():
            new_body = Body(name=PrefixedName("b3"))
            w1.add_kinematic_structure_entity(new_body)

        time.sleep(0.1)
        self.assertEqual(len(w1.kinematic_structure_entities), 1)
        self.assertEqual(len(w2.kinematic_structure_entities), 1)

        assert w2.get_kinematic_structure_entity_by_name("b3")

    def test_model_synchronization_creation_only(self):

        w1 = World(name="w1")
        w2 = World(name="w2")

        synchronizer_1 = WorldSynchronizer(self.node, w1, subscribe=False)
        synchronizer_2 = WorldSynchronizer(self.node, w2)

        with w1.modify_world():
            b2 = Body(name=PrefixedName("b2"))
            w1.add_kinematic_structure_entity(b2)

            new_body = Body(name=PrefixedName("b3"))
            w1.add_kinematic_structure_entity(new_body)

            c = Connection6DoF(b2, new_body, _world=w1)
            w1.add_connection(c)
        time.sleep(0.1)
        self.assertEqual(len(w1.kinematic_structure_entities), 2)
        self.assertEqual(len(w2.kinematic_structure_entities), 2)
        self.assertEqual(len(w1.connections), 1)
        self.assertEqual(len(w2.connections), 1)


    @classmethod
    def tearDownClass(cls):
        # cls.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
