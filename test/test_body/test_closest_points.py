import pytest

from semantic_world.connections import Connection6DoF
from semantic_world.geometry import Box, Scale, Sphere
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types.spatial_types import TransformationMatrix
from semantic_world.world import World
from semantic_world.world_entity import Body

@pytest.fixture
def world_setup_simple():
    world = World()
    root = Body(PrefixedName(name='root', prefix='world'))
    body1 = Body(PrefixedName('name1', prefix='test'), collision=[Sphere(origin=TransformationMatrix.from_xyz_rpy(), radius=0.01)])
    body2 = Body(PrefixedName('name2', prefix='test'), collision=[Sphere(origin=TransformationMatrix.from_xyz_rpy(), radius=0.01)])

    with world.modify_world():
        world.add_body(body1)
        world.add_body(body2)

        c_root_body1 = Connection6DoF(parent=root, child=body1, _world=world)
        c_root_body2 = Connection6DoF(parent=root, child=body2, _world=world)
        world.add_connection(c_root_body1)
        world.add_connection(c_root_body2)


    return world, body1, body2

def test_closest_points(world_setup_simple):
    world, body1, body2 = world_setup_simple

    with world.modify_world():
        world.add_body(body1)
        world.add_body(body2)


    closest_points = body1.compute_closest_points_multi([body2])
    assert closest_points[0] != closest_points[1]