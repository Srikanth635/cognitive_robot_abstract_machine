import os
import unittest

import sqlalchemy
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.orm.ormatic_interface import *
from krrood.ormatic.dao import to_dao


import pytest

urdf_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "urdf"
)
table_path = os.path.join(urdf_dir, "table.urdf")


@pytest.fixture
def engine():
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def table_world():
    return URDFParser.from_file(file_path=table_path).parse()


@pytest.fixture
def session(engine):
    session = Session(engine)
    Base.metadata.create_all(bind=session.bind)
    yield session
    Base.metadata.drop_all(session.bind)
    session.close()


def test_table_world(session, table_world):
    world_dao: WorldMappingDAO = to_dao(table_world)

    session.add(world_dao)
    session.commit()

    bodies_from_db = session.scalars(select(BodyDAO)).all()
    assert len(bodies_from_db) == len(table_world.kinematic_structure_entities)

    connections_from_db = session.scalars(select(ConnectionDAO)).all()
    assert len(connections_from_db) == len(table_world.connections)

    queried_world = session.scalar(select(WorldMappingDAO))
    reconstructed = queried_world.from_dao()


def test_insert(session):
    origin = TransformationMatrix.from_xyz_rpy(1, 2, 3, 1, 2, 3)
    scale = Scale(1.0, 1.0, 1.0)
    color = Color(0.0, 1.0, 1.0)
    shape1 = Box(origin=origin, scale=scale, color=color)
    b1 = Body(name=PrefixedName("b1"), collision=ShapeCollection([shape1]))

    dao: BodyDAO = to_dao(b1)
    assert dao.collision.shapes[0].origin is not None

    session.add(dao)
    session.commit()
    queried_body = session.scalar(select(BodyDAO))
    assert queried_body.collision.shapes[0].origin is not None
    reconstructed_body = queried_body.from_dao()
    assert reconstructed_body is reconstructed_body.collision[0].origin.reference_frame

    result = session.scalar(select(ShapeDAO))
    assert isinstance(result, BoxDAO)
    box = result.from_dao()
