import os
import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.adapters.fbx import FBXParser
from semantic_world.utils import drop_database
from semantic_world.orm.ormatic_interface import *

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Generate world from FBX file and store in database')

# Default FBX path
default_fbx_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fbx", "shelves_group.fbx")
parser.add_argument('--fbx-path', type=str, default=default_fbx_path,
                    help=f'Path to the FBX file (default: {default_fbx_path})')

# Default database URI from environment variable
default_db_uri = os.environ.get("SEMANTIC_WORLD_DATABASE_URI", "")
parser.add_argument('--db-uri', type=str, default=default_db_uri,
                    help='Database URI (default: value from SEMANTIC_WORLD_DATABASE_URI environment variable)')

# Option to drop database
parser.add_argument('--drop-db', action='store_true', default=True,
                    help='Drop the database before creating new tables (default: True)')
parser.add_argument('--no-drop-db', action='store_false', dest='drop_db',
                    help='Do not drop the database before creating new tables')

# Parse arguments
args = parser.parse_args()

# Use the parsed arguments
fbx_path = args.fbx_path
semantic_world_database_uri = args.db_uri

# Validate database URI
if not semantic_world_database_uri:
    raise ValueError("Database URI must be provided either via --db-uri argument or SEMANTIC_WORLD_DATABASE_URI environment variable")

# Create database engine and session
engine = create_engine(f'mysql+pymysql://{semantic_world_database_uri}')
session = Session(engine)

# Drop database if requested
if args.drop_db:
    drop_database(engine)
    
Base.metadata.create_all(engine)

parser = FBXParser(fbx_path)
worlds = parser.parse()

daos = [to_dao(world) for world in worlds]
session.add_all(daos)
session.commit()

