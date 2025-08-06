import os
import re

import tqdm

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.adapters.fbx import FBXParser
from semantic_world.orm.ormatic_interface import *
from semantic_world.utils import drop_database


def parse_fbx_file(fbx_file, session):
    parser = FBXParser(fbx_file)
    worlds = parser.parse()
    daos = [to_dao(world) for world in worlds]
    session.add_all(daos)


def main():
    semantic_world_database_uri = os.environ.get("SEMANTIC_WORLD_DATABASE_URI")
    procthor_root = os.path.join(os.path.expanduser("~"), "ai2thor")

    files = []
    for root, dirs, filenames in os.walk(procthor_root):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    pattern = re.compile(r'.*_grp\.fbx$', re.IGNORECASE)

    excluded_words = ["FirstPersonCharacter", "SourceFiles_Procedural", "RobotArmTest", "_shards_"]

    fbx_files = [f for f in files if not any([e in f for e in excluded_words]) and pattern.fullmatch(f)]
    # Create database engine and session
    engine = create_engine(f'mysql+pymysql://{semantic_world_database_uri}')
    session = Session(engine)

    # update schema
    drop_database(engine)
    Base.metadata.create_all(engine)

    for fbx_file in tqdm.tqdm(fbx_files):
        parse_fbx_file(fbx_file, session)
    session.commit()

if __name__ == '__main__':
    main()
