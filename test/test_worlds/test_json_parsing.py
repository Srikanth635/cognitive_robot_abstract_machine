import os
import unittest

import trimesh.boolean

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.world_description.geometry import Box
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def test_json_serialization():
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
