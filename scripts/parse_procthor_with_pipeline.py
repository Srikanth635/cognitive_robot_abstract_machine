import os
import re

import tqdm

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ormatic.dao import to_dao
from semantic_world.adapters.factories import HandleFactory, ContainerFactory, Direction, DrawerFactory, DoorFactory, \
    DresserFactory
from semantic_world.adapters.fbx import FBXParser
from semantic_world.orm.ormatic_interface import *
from semantic_world.pipeline.pipeline import Pipeline, COACDMeshDecomposer, BodyFilter, BodyFactoryReplace
from semantic_world.prefixed_name import PrefixedName
from semantic_world.utils import drop_database
from semantic_world.world_entity import Body


def dresser_factory_replace(dresser: Body) -> DresserFactory:

    drawer_pattern = re.compile(r'^.*_drawer_.*$')
    door_pattern = re.compile(r'^.*_door_.*$')

    drawer_factories = []
    drawer_transforms = []
    door_factories = []
    door_transforms = []
    for child in dresser.child_bodies:
        if bool(drawer_pattern.fullmatch(child.name.name)):
            drawer_transforms.append(child.parent_connection.origin_expression)

            handle_factory = HandleFactory(name=PrefixedName(child.name.name + "_handle", child.name.prefix),
                                           width=0.1)
            container_factory = ContainerFactory(
                name=PrefixedName(child.name.name + "_container", child.name.prefix),
                scale=child.as_bounding_box_collection(dresser._world.root).bounding_boxes[0].scale, direction=Direction.Z)
            drawer_factory = DrawerFactory(name=child.name, handle_factory=handle_factory,
                                           container_factory=container_factory)
            drawer_factories.append(drawer_factory)
        elif bool(door_pattern.fullmatch(child.name.name)):
            door_transforms.append(child.parent_connection.origin_expression)
            handle_factory = HandleFactory(PrefixedName(child.name.name + "_handle", child.name.prefix), 0.1)

            door_factory = DoorFactory(name=child.name, scale=child.as_bounding_box_collection(dresser._world.root).bounding_boxes[0].scale,
                                       handle_factory=handle_factory,
                                       handle_direction=Direction.Y)
            door_factories.append(door_factory)

    dresser_container_factory = ContainerFactory(
        name=PrefixedName(dresser.name.name + "_container", dresser.name.prefix),
        scale=dresser.as_bounding_box_collection(dresser._world.root).bounding_boxes[0].scale, direction=Direction.X)
    dresser_factory = DresserFactory(name=dresser.name, container_factory=dresser_container_factory,
                                     drawers_factories=drawer_factories,
                                     drawer_transforms=drawer_transforms, door_factories=door_factories,
                                     door_transforms=door_transforms)

    return dresser_factory

def parse_fbx_file(fbx_file, session):
    dresser_pattern = re.compile(r'^dresser_\d+.*$')
    pipeline = Pipeline(
        [
            BodyFilter(lambda x: not x.name.name.startswith("PS_")),
            # COACDMeshDecomposer(search_iterations=60, max_convex_hull=1)
            BodyFactoryReplace(
                body_condition=lambda b: bool(dresser_pattern.fullmatch(b.name.name)),
                factory_creator=dresser_factory_replace,
            )
         ]
    )
    parser = FBXParser(fbx_file)
    worlds = parser.parse()

    worlds = [pipeline.apply(world) for world in worlds]

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
        if "dressers_grp" in fbx_file:
            parse_fbx_file(fbx_file, session)
    session.commit()

if __name__ == '__main__':
    main()
