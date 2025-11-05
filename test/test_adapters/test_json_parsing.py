import os

import numpy as np
import trimesh.boolean

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_body_json_serialization():
    body = Body(name=PrefixedName("body"))
    collision = [Box(origin=TransformationMatrix.from_xyz_rpy(0, 1, 0, 0, 0, 1, body))]
    body.collision = ShapeCollection(collision, reference_frame=body)

    json_data = body.to_json()
    body2 = Body.from_json(json_data)

    for c1 in body.collision:
        for c2 in body2.collision:
            assert c1 == c2

    assert (
        body.collision.shapes[0].origin.reference_frame
        == body2.collision.shapes[0].origin.reference_frame
    )

    assert (
        body.collision.shapes[0].origin.child_frame
        == body2.collision.shapes[0].origin.child_frame
    )

    assert id(body.collision.shapes[0].origin.reference_frame) != id(
        body2.collision.shapes[0].origin.reference_frame
    )

    assert body == body2


def test_transformation_matrix_json_serialization():
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    t = TransformationMatrix(reference_frame=body, child_frame=body2)
    json_data = t.to_json()
    kwargs = {}
    tracker = WorldEntityKwargsTracker.from_kwargs(kwargs)
    tracker.add_parsed_world_entity(body)
    tracker.add_parsed_world_entity(body2)
    t2 = TransformationMatrix.from_json(json_data, **kwargs)
    assert t.reference_frame == t2.reference_frame
    assert id(t.reference_frame) == id(t2.reference_frame)


def test_connection_json_serialization_with_world():
    world = World()
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_kinematic_structure_entity(body2)
        c = FixedConnection(
            parent=body,
            child=body2,
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                x=1, roll=2, reference_frame=body, child_frame=body2
            ),
        )
        world.add_connection(c)
    json_data = c.to_json()
    c2 = FixedConnection.from_json(json_data, world=world)
    assert c == c2
    assert np.allclose(
        c.parent_T_connection_expression.to_np(),
        c2.parent_T_connection_expression.to_np(),
    )
    assert (
        c.parent_T_connection_expression.reference_frame
        == c2.parent_T_connection_expression.reference_frame
    )
    assert (
        c.parent_T_connection_expression.child_frame
        == c2.parent_T_connection_expression.child_frame
    )


def test_transformation_matrix_json_serialization_with_world_in_kwargs():
    world = World()
    body = Body(name=PrefixedName("body"))
    body2 = Body(name=PrefixedName("body2"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_kinematic_structure_entity(body2)
        c = FixedConnection(
            parent=body,
            child=body2,
            parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(
                x=1, roll=2, reference_frame=body, child_frame=body2
            ),
        )
        world.add_connection(c)
    json_data = c.to_json()
    kwargs = {"world": world}
    WorldEntityKwargsTracker.from_kwargs(kwargs)
    c2 = FixedConnection.from_json(json_data, **kwargs)
    assert c == c2
    assert np.allclose(
        c.parent_T_connection_expression.to_np(),
        c2.parent_T_connection_expression.to_np(),
    )
    assert (
        c.parent_T_connection_expression.reference_frame
        == c2.parent_T_connection_expression.reference_frame
    )
    assert (
        c.parent_T_connection_expression.child_frame
        == c2.parent_T_connection_expression.child_frame
    )


def test_json_serialization_with_mesh():
    body: Body = (
        STLParser(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "resources",
                "stl",
                "milk.stl",
            )
        )
        .parse()
        .root
    )

    json_data = body.to_json()
    body2 = Body.from_json(json_data)

    for c1 in body.collision:
        for c2 in body2.collision:
            assert (trimesh.boolean.difference([c1.mesh, c2.mesh])).is_empty
