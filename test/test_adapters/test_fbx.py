import os
import time
import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.orm.ormatic_interface import *
from semantic_world.adapters.fbx import FBXParser

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
        world = parser.parse()

        dao = to_dao(world)
        self.session.add(dao)
        self.session.commit()

    def test_query(self):
        from semantic_world.adapters.viz_marker import VizMarkerPublisher

        query = select(WorldMappingDAO)
        world_dao = self.session.scalars(query).first()
        world = world_dao.from_dao()



        import rclpy
        rclpy.init()
        node = rclpy.create_node("viz_marker")

        p = VizMarkerPublisher(world, node)
        time.sleep(100)