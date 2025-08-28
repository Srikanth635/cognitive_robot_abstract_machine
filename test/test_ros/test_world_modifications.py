import unittest

from semantic_world.connections import Connection6DoF
from semantic_world.prefixed_name import PrefixedName
from semantic_world.world import World
from semantic_world.world_entity import Body
from semantic_world.world_modification import AddConnectionModification


class ConnectionModificationTestCase(unittest.TestCase):

    def test_something(self):
        w = World()

        with w.modify_world():
            b1 = Body(name=PrefixedName("b1"))
            b2 = Body(name=PrefixedName("b2"))
            w.add_kinematic_structure_entity(b1)
            w.add_kinematic_structure_entity(b2)

            connection = Connection6DoF(b1, b2, _world=w)
            w.add_connection(connection)

        connection = w.connections[0]
        modification = AddConnectionModification.from_connection(connection)
        print(modification.to_json())


if __name__ == "__main__":
    unittest.main()
