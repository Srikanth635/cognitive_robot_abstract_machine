"""
Tests of what a scene bundle states a container holds: the stated contents are read for
the container's body, and every rewrite or variant of the bundle keeps them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from llm_reasoner.knowledge.access import pouring_frame

from cramera.resolution.resolver.container_contents import (
    ContainerContents,
    ContentsKey,
)
from cramera.resolution.resolver.exceptions import (
    FillFractionOutOfRange,
    UnknownMaterialRegime,
)
from cramera.resolution.resolver.overturned_scene import OverturnedScene
from cramera.resolution.resolver.scene_world import SCENE_FILE, SceneKey, SceneObjects
from cramera.resolution.scene_conversion import SceneArchiveConversion

from .test_scene_conversion import COMMITTED_ARCHIVE, LEGACY_ARCHIVE

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


# %% stated contents


class TestStatedContents:
    def test_contents_read_back_as_written(self):
        contents = ContainerContents(
            material_regime=pouring_frame.MaterialRegime.GRANULAR, fill_fraction=0.25
        )
        assert ContainerContents.from_json(contents.to_json()) == contents

    def test_a_fill_fraction_above_full_is_refused(self):
        with pytest.raises(FillFractionOutOfRange):
            ContainerContents(
                material_regime=pouring_frame.MaterialRegime.LIQUID, fill_fraction=1.5
            )

    def test_a_negative_fill_fraction_is_refused(self):
        with pytest.raises(FillFractionOutOfRange):
            ContainerContents.from_json(
                {
                    ContentsKey.MATERIAL_REGIME: pouring_frame.MaterialRegime.LIQUID.name,
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
        objects = SceneObjects.read(COMMITTED_ARCHIVE / "simple_pouring")
        assert objects.contents_of(PITCHER_BODY) == ContainerContents(
            material_regime=pouring_frame.MaterialRegime.LIQUID, fill_fraction=0.5
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

    def test_converting_the_archive_keeps_the_contents(self, pouring_bundle, tmp_path):
        archive = tmp_path / "archive"
        shutil.copytree(LEGACY_ARCHIVE, archive)
        shutil.copytree(pouring_bundle, archive / pouring_bundle.name)
        converted = tmp_path / "converted"

        SceneArchiveConversion(source=archive, target=converted).convert()

        assert SceneObjects.read(converted / pouring_bundle.name).contents_of(
            PITCHER_BODY
        ) == stated_contents(pouring_bundle)
