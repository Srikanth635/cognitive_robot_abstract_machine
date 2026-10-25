"""
Tests of what a scene bundle states a container holds: the stated contents are read for
the container's body, and every rewrite or variant of the bundle keeps them.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from llm_reasoner.knowledge.access import action_core

from cram_viz import paths
from cram_viz.onboard.demo import Recorder, build_scene
from cram_viz.resolution.container_contents import ContainerContents, ContentsKey
from cram_viz.resolution.exceptions import (
    FillFractionOutOfRange,
    UnknownMaterialRegime,
)
from cram_viz.resolution.overturned_scene import OverturnedScene
from cram_viz.resolution.scene_world import SCENE_FILE, SceneKey, SceneObjects

PITCHER_BODY = "Static_MilkPitcher.stl"
"""
The body of the container the pouring scenes state contents for.
"""

# %% helpers


def stated_contents(bundle: Path) -> ContainerContents:
    """
    :return: The contents the bundle's scene description states for the pitcher.
    """
    scene = json.loads((bundle / SCENE_FILE).read_text())
    (pitcher,) = [
        entry
        for entry in scene[SceneKey.OBJECTS]
        if Path(entry[SceneKey.MESH]).name == PITCHER_BODY
    ]
    return ContainerContents.from_json(pitcher[SceneKey.CONTENTS])


@dataclass
class NamedBody:
    """
    A body known only by its name.
    """

    name: str
    """
    The body's name.
    """


@dataclass
class RobotWithoutArms:
    """
    A robot whose parts are only its root body.
    """

    root: NamedBody = field(default_factory=lambda: NamedBody(name="base_footprint"))
    """
    The robot's root body.
    """

    def get_arms(self) -> list:
        """
        :return: No arms.
        """
        return []


@dataclass
class WorldWithoutBodies:
    """
    A world holding no bodies.
    """

    bodies: list = field(default_factory=list)
    """
    The world's bodies.
    """


def recording_of_still_pitcher(mesh_source: Path) -> Recorder:
    """
    :param mesh_source: The pitcher's mesh file, named for its body.
    :return: A one-frame recording in which only the pitcher stands still.
    """
    recorder = Recorder()
    recorder.frames = [{}]
    recorder.base_frames = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
    recorder.obj_frames = [{PITCHER_BODY: [2.37, 1.6, 1.02, 0.0, 0.0, 0.0, 1.0]}]
    recorder.stl_sources = [str(mesh_source)]
    recorder.robot = RobotWithoutArms()
    recorder.world = WorldWithoutBodies()
    return recorder


# %% stated contents


class TestStatedContents:
    def test_contents_read_back_as_written(self):
        contents = ContainerContents(
            material_regime=action_core.MaterialRegime.GRANULAR, fill_fraction=0.25
        )
        assert ContainerContents.from_json(contents.to_json()) == contents

    def test_a_fill_fraction_above_full_is_refused(self):
        with pytest.raises(FillFractionOutOfRange):
            ContainerContents(
                material_regime=action_core.MaterialRegime.LIQUID, fill_fraction=1.5
            )

    def test_a_negative_fill_fraction_is_refused(self):
        with pytest.raises(FillFractionOutOfRange):
            ContainerContents.from_json(
                {
                    ContentsKey.MATERIAL_REGIME: action_core.MaterialRegime.LIQUID.name,
                    ContentsKey.FILL_FRACTION: -0.1,
                }
            )

    def test_a_regime_the_ontology_does_not_name_is_refused(self):
        with pytest.raises(UnknownMaterialRegime):
            ContainerContents.from_json(
                {
                    ContentsKey.MATERIAL_REGIME: "Plasma",
                    ContentsKey.FILL_FRACTION: 0.5,
                }
            )


# %% contents of a body


class TestContentsOfBody:
    def test_the_simple_pouring_pitcher_is_half_full_of_liquid(self):
        objects = SceneObjects.read(paths.BUNDLED_SCENES / "simple_pouring")
        assert objects.contents_of(PITCHER_BODY) == ContainerContents(
            material_regime=action_core.MaterialRegime.LIQUID, fill_fraction=0.5
        )

    def test_a_body_whose_entry_states_no_contents_has_none(self, pouring_bundle):
        assert SceneObjects.read(pouring_bundle).contents_of("bowl.stl") is None

    def test_a_body_the_scene_does_not_describe_has_none(self, pouring_bundle):
        assert SceneObjects.read(pouring_bundle).contents_of("table.stl") is None


# %% rewrites of the bundle


class TestContentsKept:
    def test_an_overturned_scene_keeps_the_contents(self, pouring_bundle, tmp_path):
        overturned = OverturnedScene(bundle=pouring_bundle, identifier="bowl").write_to(
            tmp_path / "overturned"
        )
        assert SceneObjects.read(overturned).contents_of(
            PITCHER_BODY
        ) == stated_contents(pouring_bundle)

    def test_onboarding_again_keeps_the_contents(self, pouring_bundle, tmp_path):
        recording = tmp_path / "recording"
        recording.mkdir()
        mesh_source = recording / PITCHER_BODY
        shutil.copy(pouring_bundle / "meshes" / "objects" / PITCHER_BODY, mesh_source)
        stated = stated_contents(pouring_bundle)

        build_scene(
            recording_of_still_pitcher(mesh_source),
            "pouring",
            str(pouring_bundle),
            step=1,
        )

        assert SceneObjects.read(pouring_bundle).contents_of(PITCHER_BODY) == stated
