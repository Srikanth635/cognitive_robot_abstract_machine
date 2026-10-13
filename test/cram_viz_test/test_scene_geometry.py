"""
Tests of the scene's geometry as the resolver reads it: every annotated object's origin,
bounding box and top in the world frame, and where the robot's base stands, stated as a
fact of the situation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from llm_reasoner.resolver.prompt_format import PromptSection
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody

from cram_viz.resolution.scene_geometry import (
    COORDINATE_DIGITS,
    GEOMETRY_HEADING,
    GeometryKey,
    SceneGeometry,
)
from cram_viz.resolution.scene_world import SceneKey, SceneWorldBuilder

from .test_scene_world import recorded_start

TOLERANCE = 10.0**-COORDINATE_DIGITS
"""
How far a stated coordinate may lie from the exact one: its rounding.
"""

# %% helpers


def described(bundle: Path) -> dict:
    """
    :return: The geometry of the bundle's scene, as the resolver reads it.
    """
    return json.loads(
        SceneGeometry(world=SceneWorldBuilder().build(bundle)).as_json_text()
    )


def object_named(geometry: dict, name: str) -> dict:
    """
    :return: The geometry of the object whose body has the given name.
    """
    (entry,) = [
        entry
        for entry in geometry[GeometryKey.OBJECTS]
        if entry[GeometryKey.NAME] == name
    ]
    return entry


def spawned(bundle: Path, identifier: str) -> list:
    """
    :return: Where the scene spawns the object of the given identifier.
    """
    scene = json.loads((bundle / "scene.json").read_text())
    (entry,) = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if entry[SceneKey.IDENTIFIER] == identifier
    ]
    return entry[SceneKey.SPAWN][:3]


# %% the objects


class TestObjects:
    def test_every_annotated_object_is_described_by_its_kind(self, pouring_bundle):
        world = SceneWorldBuilder().build(pouring_bundle)
        geometry = json.loads(SceneGeometry(world=world).as_json_text())
        assert sorted(
            (entry[GeometryKey.KIND], entry[GeometryKey.NAME])
            for entry in geometry[GeometryKey.OBJECTS]
        ) == sorted(
            (type(annotation).__name__, annotation.root.name.name)
            for annotation in world.get_semantic_annotations_by_type(HasRootBody)
        )

    def test_an_object_stands_at_its_origin(self, pouring_bundle):
        pitcher = object_named(described(pouring_bundle), "Static_MilkPitcher.stl")
        assert np.allclose(
            pitcher[GeometryKey.ORIGIN],
            spawned(pouring_bundle, "Static_MilkPitcher"),
            atol=TOLERANCE,
        )

    def test_the_top_is_the_centre_of_the_bounding_box_top(self, pouring_bundle):
        bowl = object_named(described(pouring_bundle), "bowl.stl")
        lowest = np.array(bowl[GeometryKey.BOUNDING_BOX][GeometryKey.MINIMUM])
        highest = np.array(bowl[GeometryKey.BOUNDING_BOX][GeometryKey.MAXIMUM])
        assert np.allclose(
            bowl[GeometryKey.TOP_CENTRE],
            [*(0.5 * (lowest[:2] + highest[:2])), highest[2]],
            atol=TOLERANCE,
        )


# %% the robot


class TestRobotBase:
    def test_the_robot_base_stands_where_the_recording_starts(self, pouring_bundle):
        (base,) = described(pouring_bundle)[GeometryKey.ROBOT_BASES]
        assert np.allclose(
            base[GeometryKey.POSITION],
            recorded_start(pouring_bundle)[:2],
            atol=TOLERANCE,
        )


# %% in the prompt


class TestSection:
    def test_the_geometry_is_a_fact_of_the_situation(self, pouring_bundle):
        geometry = SceneGeometry(world=SceneWorldBuilder().build(pouring_bundle))
        assert geometry.section is PromptSection.SITUATION

    def test_the_geometry_is_stated_with_its_frame_and_units(self, pouring_bundle):
        geometry = SceneGeometry(world=SceneWorldBuilder().build(pouring_bundle))
        text = geometry.context_for(None)
        assert text.startswith(GEOMETRY_HEADING)
        assert text.endswith(geometry.as_json_text())
