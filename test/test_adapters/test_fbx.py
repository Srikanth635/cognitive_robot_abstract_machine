import os
import time
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.adapters.fbx import FBXParser
from semantic_world.adapters.procthor import replace_dresser_drawer_connections
from semantic_world.connections import PrismaticConnection
from semantic_world.orm.ormatic_interface import *
from semantic_world.reasoner import WorldReasoner
from semantic_world.world import World


class FBXParserTest(unittest.TestCase):
    fbx_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fbx", "dressers_group.fbx")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.engine = create_engine('mysql+pymysql://semantic_world@localhost:3306/semantic_world')

    def setUp(self):
        super().setUp()
        self.session = Session(self.engine)
        Base.metadata.create_all(bind=self.session.bind)

    def test_parse_fbx(self):
        # Base.metadata.drop_all(bind=self.session.bind)
        parser = FBXParser(self.fbx_path)
        worlds = parser.parse()

        daos = [to_dao(world) for world in worlds]
        self.session.add_all(daos)
        self.session.commit()

    def test_query(self):
        from semantic_world.adapters.viz_marker import VizMarkerPublisher
        import rclpy
        rclpy.init()

        query = select(WorldMappingDAO)
        world_dao = self.session.scalars(query).all()
        world: World = world_dao[0].from_dao()

        node = rclpy.create_node("viz_marker")

        p = VizMarkerPublisher(world, node)
        time.sleep(100)
        p._stop_publishing()
        rclpy.shutdown()

    def test_rdr_creation(self):
        from semantic_world.adapters.viz_marker import VizMarkerPublisher
        import rclpy
        rclpy.init()
        node = rclpy.create_node("viz_marker")

        query = select(WorldMappingDAO)
        world_dao = self.session.scalars(query).all()
        world: World = world_dao[0].from_dao()

        replace_dresser_drawer_connections(world)
        prismatic_connection = list(filter(lambda x: isinstance(x, PrismaticConnection), world.connections))
        self.assertEqual(len(prismatic_connection), 4)

        world.state[prismatic_connection[0].dof.name].position = 2
        world.notify_state_change()
        p = VizMarkerPublisher(world, node)
        time.sleep(100)
        p._stop_publishing()
        rclpy.shutdown()