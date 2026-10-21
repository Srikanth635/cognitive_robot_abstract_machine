"""
Tests the conversion of scene bundles in the format cram_viz writes into the format
cramera reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cramera import paths as cramera_paths
from urdf_parser_py.urdf import URDF
from urdf_parser_py.xml_reflection.core import ParseError

from cramera_resolution.exceptions import (
    SceneArchiveHasNoIndex,
    TargetInsideSceneArchive,
)
from cramera_resolution.scene_conversion import SceneArchiveConversion

LEGACY_ARCHIVE = Path(__file__).parent / "scene_conversion_cases" / "legacy"
COMMITTED_ARCHIVE = cramera_paths.SCENES_SUBMODULE

# %% fixtures


@pytest.fixture()
def converted(tmp_path) -> Path:
    """
    The legacy archive converted.
    """
    target = tmp_path / "converted"
    SceneArchiveConversion(source=LEGACY_ARCHIVE, target=target).convert()
    return target


# %% the place target
class TestPlaceTarget:
    def test_the_position_is_stored_as_position(self, converted):
        scene = json.loads((converted / "tiny" / "scene.json").read_text())
        assert scene["placeTarget"]["position"] == [1.5, 2.5]

    def test_the_old_key_is_gone(self, converted):
        scene = json.loads((converted / "tiny" / "scene.json").read_text())
        assert "pos" not in scene["placeTarget"]

    def test_the_rest_of_the_place_target_is_kept(self, converted):
        legacy = json.loads((LEGACY_ARCHIVE / "tiny" / "scene.json").read_text())
        scene = json.loads((converted / "tiny" / "scene.json").read_text())
        assert {
            key: value
            for key, value in scene["placeTarget"].items()
            if key != "position"
        } == {
            key: value for key, value in legacy["placeTarget"].items() if key != "pos"
        }

    def test_the_rest_of_the_scene_is_kept(self, converted):
        legacy = json.loads((LEGACY_ARCHIVE / "tiny" / "scene.json").read_text())
        scene = json.loads((converted / "tiny" / "scene.json").read_text())
        assert {k: v for k, v in scene.items() if k != "placeTarget"} == {
            k: v for k, v in legacy.items() if k != "placeTarget"
        }

    def test_a_scene_without_a_place_target_is_unchanged(self, converted):
        assert json.loads(
            (converted / "bare" / "scene.json").read_text()
        ) == json.loads((LEGACY_ARCHIVE / "bare" / "scene.json").read_text())


# %% the robot description
class TestTransmissions:
    def test_the_converted_description_is_read_by_the_urdf_parser(self, converted):
        robot = URDF.from_xml_string((converted / "tiny" / "robot.urdf").read_text())
        assert [link.name for link in robot.links] == ["base", "arm"]

    def test_the_original_description_is_not(self):
        with pytest.raises(ParseError):
            URDF.from_xml_string((LEGACY_ARCHIVE / "tiny" / "robot.urdf").read_text())

    def test_only_the_transmissions_are_removed(self, converted):
        legacy = (LEGACY_ARCHIVE / "tiny" / "robot.urdf").read_text()
        text = (converted / "tiny" / "robot.urdf").read_text()
        assert text == legacy[: legacy.index("  <transmission")] + "</robot>\n"


# %% the rest of the archive
class TestArchive:
    @pytest.mark.parametrize(
        "relative", ["index.json", "tiny/trajectory.json", "tiny/meshes/part.stl"]
    )
    def test_other_files_are_copied_as_they_are(self, converted, relative):
        assert (converted / relative).read_bytes() == (
            LEGACY_ARCHIVE / relative
        ).read_bytes()

    def test_converting_a_converted_archive_changes_nothing(self, converted, tmp_path):
        again = tmp_path / "again"
        SceneArchiveConversion(source=converted, target=again).convert()
        assert [
            (path.relative_to(again), path.read_bytes())
            for path in sorted(again.rglob("*"))
            if path.is_file()
        ] == [
            (path.relative_to(converted), path.read_bytes())
            for path in sorted(converted.rglob("*"))
            if path.is_file()
        ]

    def test_an_archive_with_no_index_cannot_be_converted(self, tmp_path):
        with pytest.raises(SceneArchiveHasNoIndex):
            SceneArchiveConversion(source=tmp_path, target=tmp_path / "out").convert()

    def test_a_target_inside_the_source_is_refused(self):
        with pytest.raises(TargetInsideSceneArchive):
            SceneArchiveConversion(
                source=LEGACY_ARCHIVE, target=LEGACY_ARCHIVE / "tiny"
            ).convert()


# %% the committed archive
class TestCommittedArchive:
    def scene_names(self) -> list[str]:
        return json.loads((COMMITTED_ARCHIVE / "index.json").read_text())["scenes"]

    def test_every_bundle_indexed_holds_a_scene(self):
        assert all(
            (COMMITTED_ARCHIVE / name / "scene.json").is_file()
            for name in self.scene_names()
        )

    def test_every_place_target_is_in_the_format_cramera_reads(self):
        place_targets = [
            json.loads((COMMITTED_ARCHIVE / name / "scene.json").read_text()).get(
                "placeTarget"
            )
            for name in self.scene_names()
        ]
        assert all(
            "pos" not in target and len(target["position"]) == 2
            for target in place_targets
            if target
        )

    def test_every_robot_description_is_read_by_the_urdf_parser(self):
        descriptions = [
            path
            for name in self.scene_names()
            for path in (COMMITTED_ARCHIVE / name).glob("*.urdf")
        ]
        assert descriptions
        assert all(
            URDF.from_xml_string(path.read_text()).links for path in descriptions
        )
