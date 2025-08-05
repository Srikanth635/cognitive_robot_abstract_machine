import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.adapters.fbx import FBXParser
from semantic_world.utils import drop_database
from semantic_world.orm.ormatic_interface import *

fbx_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fbx", "shelves_group.fbx")

semantic_world_database_uri = os.environ["SEMANTIC_WORLD_DATABASE_URI"]
engine = create_engine(f'mysql+pymysql://{semantic_world_database_uri}')
session = Session(engine)

drop_database(engine)
Base.metadata.create_all(engine)

parser = FBXParser(fbx_path)
worlds = parser.parse()

daos = [to_dao(world) for world in worlds]
session.add_all(daos)
session.commit()

