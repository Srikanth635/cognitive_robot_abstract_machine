import unittest
from dataclasses import dataclass

from semantic_world.datastructures.prefixed_name import PrefixedName
from semantic_world.pipeline.pipeline import Step, Pipeline, BodyFilter
from semantic_world.spatial_types.spatial_types import TransformationMatrix
from semantic_world.world import World
from semantic_world.world_description.connections import FixedConnection
from semantic_world.world_description.world_entity import Body


class PipelineTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dummy_world = World()
        b1 = Body(name=PrefixedName("body1", "test"))
        b2 = Body(name=PrefixedName("body2", "test"))
        c1 = FixedConnection(b1, b2, TransformationMatrix())
        with cls.dummy_world.modify_world():
            cls.dummy_world.add_body(b1)
            cls.dummy_world.add_body(b2)
            cls.dummy_world.add_connection(c1)

    def test_pipeline_and_step(self):

        @dataclass
        class TestStep(Step):
            body_name: PrefixedName

            def _apply(self, world: World) -> World:
                b1 = Body(name=self.body_name)
                world.add_body(b1)
                return world

        pipeline = Pipeline(steps=[TestStep(body_name=PrefixedName("body1", "test"))])

        dummy_world = World()

        dummy_world = pipeline.apply(dummy_world)

        self.assertEqual(len(dummy_world.bodies), 1)
        self.assertEqual(dummy_world.root.name, PrefixedName("body1", "test"))

    def test_body_filter(self):

        pipeline = Pipeline(
            steps=[BodyFilter(lambda x: x.name == PrefixedName("body1", "test"))]
        )

        filtered_world = pipeline.apply(self.dummy_world)
        self.assertEqual(len(filtered_world.bodies), 1)
        self.assertEqual(filtered_world.root.name, PrefixedName("body1", "test"))

    @unittest.skip("Not sure how we want to test this. Add a test fbx file to repo?")
    def test_center_local_geometry_and_preserve_world_pose(self): ...

    @unittest.skip("If we load a test fbx file anyways, might as well use it here too")
    def test_body_factory_replace(self): ...

    @unittest.skip("If we load a test fbx file anyways, might as well use it here too")
    def test_dresser_factory_from_body(self): ...


if __name__ == "__main__":
    unittest.main()
