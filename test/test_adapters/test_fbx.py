import os
import time
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.adapters.fbx import FBXParser
from semantic_world.views.factories import replace_dresser_drawer_connections, HandleFactory, Direction, DoorFactory
from semantic_world.connections import RevoluteConnection
from semantic_world.geometry import Scale
from semantic_world.orm.ormatic_interface import *
from semantic_world.prefixed_name import PrefixedName
from semantic_world.world import World


class FBXParserTest(unittest.TestCase):
    fbx_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "fbx", "shelves_group.fbx")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.engine = create_engine('mysql+pymysql://semantic_world@localhost:3306/semantic_world')

    def setUp(self):
        super().setUp()
        self.session = Session(self.engine)
        Base.metadata.create_all(bind=self.session.bind)

    def test_parse_fbx(self):
        #drop_database(self.engine)
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
        world: World = world_dao[1].from_dao()

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

        handle_factory = HandleFactory(width=0.1, name=PrefixedName("door_handle"))
        factory = DoorFactory(name=PrefixedName('Door'), scale=Scale(0.03, 1, 2), handle_factory=handle_factory, handle_direction=Direction.NEGATIVE_Y)

        world = factory.create()



        p = VizMarkerPublisher(world, node)
        time.sleep(100)
        p._stop_publishing()
        rclpy.shutdown()


    def test_drawer_factory_from_fbx(self):
        from semantic_world.adapters.viz_marker import VizMarkerPublisher
        import rclpy
        rclpy.init()
        node = rclpy.create_node("viz_marker")

        query = select(WorldMappingDAO)
        world_dao = self.session.scalars(query).all()
        world: World = world_dao[5].from_dao()
        dresser_factory = replace_dresser_drawer_connections(world)


        world = dresser_factory.create()
        connections = list(filter(lambda x: isinstance(x, RevoluteConnection), world.connections))
        self.assertEqual(len(connections), 4)

        world.state[connections[0].dof.name].position = -1.5
        world.notify_state_change()

        p = VizMarkerPublisher(world, node)
        time.sleep(100)
        p._stop_publishing()
        rclpy.shutdown()

