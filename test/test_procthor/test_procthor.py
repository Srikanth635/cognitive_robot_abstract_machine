import os
import time
import unittest

import rclpy
from entity_query_language import the, entity, let
from ormatic.eql_interface import eql_to_sql
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from semantic_world.adapters.procthor.procthor_parser import ProcTHORParser
from semantic_world.adapters.viz_marker import VizMarkerPublisher
from semantic_world.orm.model import WorldMapping


class ProcTHORTestCase(unittest.TestCase):
    def test_procthor_parser_test_file(self):

        json_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "resources",
            "procthor_json",
        )
        #
        # parser = ProcTHORParser(
        #     os.path.join(json_dir, "house_987654321.json")
        # )
        # world = parser.parse()

        semantic_world_database_uri = os.environ.get("SEMANTIC_WORLD_DATABASE_URI")

        # Create database engine and session
        engine = create_engine(f"mysql+pymysql://{semantic_world_database_uri}")
        session = Session(engine)

        parser = ProcTHORParser(os.path.join(json_dir, "house_1.json"), session)
        world = parser.parse()

        ...
        # rclpy.init()
        # node = rclpy.create_node("viz_marker")
        #
        # p = VizMarkerPublisher(world, node)
        #
        # time.sleep(1000)
        # p._stop_publishing()
        # rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
