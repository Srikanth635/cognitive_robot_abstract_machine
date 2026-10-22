"""
Tests of a scene variant in which one object is turned upside down where it stands: the
object states that its up axis points down, rests on the same surface as before, and
nothing else about the bundle changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl

from cramera.resolution.resolver.overturned_scene import ObjectNotInScene, OverturnedScene
from cramera.resolution.resolver.scene_geometry import GeometryKey, SceneGeometry
from cramera.resolution.resolver.scene_world import SCENE_FILE, SceneWorldBuilder

from .test_scene_geometry import TOLERANCE, object_named

BOWL = "bowl"
"""
The identifier of the object turned over in these tests.
"""

# %% helpers


def geometry_of(bundle: Path) -> dict:
    """
    :return: The geometry of the bundle's scene, as the resolver reads it.
    """
    return json.loads(
        SceneGeometry(world=SceneWorldBuilder().build(bundle)).as_json_text()
    )


@pytest.fixture
def overturned(pouring_bundle, tmp_path) -> Path:
    """
    The pouring bundle with the bowl turned upside down.
    """
    return OverturnedScene(bundle=pouring_bundle, identifier=BOWL).write_to(
        tmp_path / "overturned"
    )


# %% the variant


class TestOverturnedScene:
    def test_the_object_states_that_its_up_points_down(self, overturned):
        bowl = object_named(geometry_of(overturned), "bowl.stl")
        assert np.allclose(bowl[GeometryKey.UP], [0.0, 0.0, -1.0], atol=TOLERANCE)

    def test_the_object_rests_where_it_stood(self, pouring_bundle, overturned):
        before = object_named(geometry_of(pouring_bundle), "bowl.stl")
        after = object_named(geometry_of(overturned), "bowl.stl")
        assert after[GeometryKey.BOUNDING_BOX][GeometryKey.MINIMUM][2] == pytest.approx(
            before[GeometryKey.BOUNDING_BOX][GeometryKey.MINIMUM][2], abs=TOLERANCE
        )

    def test_every_other_object_stays_as_it_was(self, pouring_bundle, overturned):
        before = geometry_of(pouring_bundle)
        after = geometry_of(overturned)
        others = [
            entry[GeometryKey.NAME]
            for entry in before[GeometryKey.OBJECTS]
            if entry[GeometryKey.NAME] != "bowl.stl"
        ]
        assert others
        for name in others:
            assert object_named(after, name) == object_named(before, name)

    def test_the_original_bundle_is_left_alone(self, pouring_bundle, overturned):
        scene = json.loads((pouring_bundle / SCENE_FILE).read_text())
        (bowl,) = [entry for entry in scene["objects"] if entry["id"] == BOWL]
        assert bowl["spawn"][3:] == [0.0, 0.0, 0.0, 1.0]

    def test_an_object_the_scene_lacks_is_refused(self, pouring_bundle, tmp_path):
        with pytest.raises(ObjectNotInScene):
            OverturnedScene(bundle=pouring_bundle, identifier="spoon").write_to(
                tmp_path / "overturned"
            )


# %% finding the object by its body


class TestFindingTheObject:
    def test_the_body_of_an_object_is_the_one_the_world_annotates_as_it(
        self, pouring_bundle
    ):
        scene = OverturnedScene(bundle=pouring_bundle, identifier=BOWL)
        world = SceneWorldBuilder().build(pouring_bundle)
        (bowl,) = world.get_semantic_annotations_by_type(Bowl)
        assert scene.body_name == bowl.root.name.name

    def test_an_object_is_found_by_the_name_of_its_body(self, pouring_bundle):
        scene = OverturnedScene(bundle=pouring_bundle, identifier=BOWL)
        found = OverturnedScene.of_body(pouring_bundle, scene.body_name)
        assert found == scene

    def test_a_body_the_scene_lacks_is_refused(self, pouring_bundle):
        with pytest.raises(ObjectNotInScene):
            OverturnedScene.of_body(pouring_bundle, "no_such_body.stl")
