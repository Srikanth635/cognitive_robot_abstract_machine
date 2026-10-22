"""
Conversion of scene bundles written by cram_viz into the format cramera reads.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from cramera.resolution.exceptions import (
    SceneArchiveHasNoIndex,
    TargetInsideSceneArchive,
)

# %% what a bundle holds


class ArchiveFile(StrEnum):
    """
    The files of a scene archive the conversion reads or recognises.
    """

    INDEX = "index.json"
    """
    The archive's list of the scenes it holds.
    """

    SCENE = "scene.json"
    """
    The file describing one scene of the archive.
    """

    ROBOT_DESCRIPTIONS = "*.urdf"
    """
    The pattern the robot and environment descriptions of a bundle match.
    """


class SceneKey(StrEnum):
    """
    The keys of a scene description the conversion reads.
    """

    PLACE_TARGET = "placeTarget"
    """
    Where the scene's object is to be placed.
    """

    CRAM_VIZ_POSITION = "pos"
    """
    The place target's position, as cram_viz writes it.
    """

    CRAMERA_POSITION = "position"
    """
    The place target's position, as cramera reads it.
    """


class TransmissionPattern(StrEnum):
    """
    The text of a robot description that cramera's URDF reader cannot read.
    """

    ELEMENT = r"[ \t]*<transmission\b.*?</transmission>[ \t]*\n?"
    """
    One ``transmission`` element with the line it stands on.
    """


# %% the conversion


@dataclass
class SceneArchiveConversion:
    """
    Writes a copy of a scene archive in the format cramera reads.

    The place target's position is stored as ``position``, and the ``transmission``
    elements of the robot descriptions, which only describe the real robot's actuators
    and which cramera's URDF reader refuses, are left out. Every other file is copied as
    it is. Converting an archive that is already converted changes nothing.
    """

    source: Path
    """
    The archive to convert.
    """

    target: Path
    """
    The directory the converted archive is written to.
    """

    def convert(self) -> None:
        """
        Write the converted archive to :attr:`target`, replacing the files it holds.

        :raises SceneArchiveHasNoIndex: When :attr:`source` holds no scene index.
        :raises TargetInsideSceneArchive: When :attr:`target` lies inside :attr:`source`.
        """
        if not (self.source / ArchiveFile.INDEX).is_file():
            raise SceneArchiveHasNoIndex(directory=self.source)
        if self.source.resolve() in self.target.resolve().parents:
            raise TargetInsideSceneArchive(source=self.source, target=self.target)
        shutil.copytree(self.source, self.target, dirs_exist_ok=True)
        for scene_file in self.target.glob(f"*/{ArchiveFile.SCENE}"):
            self.convert_scene(scene_file)
        for description in self.target.glob(f"*/{ArchiveFile.ROBOT_DESCRIPTIONS}"):
            self.convert_robot_description(description)

    @staticmethod
    def convert_scene(scene_file: Path) -> None:
        """
        Store the place target's position of a scene as cramera reads it.

        :param scene_file: The scene description, rewritten in place.
        """
        scene = json.loads(scene_file.read_text())
        place_target = scene.get(SceneKey.PLACE_TARGET)
        if not place_target or SceneKey.CRAM_VIZ_POSITION not in place_target:
            return
        scene[SceneKey.PLACE_TARGET] = {
            (
                SceneKey.CRAMERA_POSITION if key == SceneKey.CRAM_VIZ_POSITION else key
            ): value
            for key, value in place_target.items()
        }
        scene_file.write_text(json.dumps(scene, indent=1) + "\n")

    @staticmethod
    def convert_robot_description(description: Path) -> None:
        """
        Leave the ``transmission`` elements out of a robot description.

        :param description: The description, rewritten in place.
        """
        description.write_text(
            re.sub(
                TransmissionPattern.ELEMENT,
                "",
                description.read_text(),
                flags=re.DOTALL,
            )
        )


# %% command line


def main() -> None:
    """
    Convert a scene archive: ``python -m cramera.resolution.scene_conversion SOURCE TARGET``.
    """
    parser = argparse.ArgumentParser(description=SceneArchiveConversion.__doc__)
    parser.add_argument("source", type=Path, help="the archive cram_viz wrote")
    parser.add_argument("target", type=Path, help="where the converted archive goes")
    arguments = parser.parse_args()
    SceneArchiveConversion(source=arguments.source, target=arguments.target).convert()


if __name__ == "__main__":
    main()
