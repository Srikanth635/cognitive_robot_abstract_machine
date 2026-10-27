"""
The scene bundle a request names, found as cramera finds it, and the objects it holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from cramera import paths as cramera_paths
from cramera.onboard.scene_index import merged_scene_index

from cramera.resolution.exceptions import NoSuchScene

# %% what a bundle holds


class SceneFile(StrEnum):
    """
    The files of a bundle the add-on reads.
    """

    SCENE = "scene.json"
    """
    The file describing the scene.
    """


class ObjectKind(StrEnum):
    """
    What an object of a scene is of.
    """

    OBJECT = "object"
    """
    A graspable object.
    """


@dataclass(frozen=True)
class SceneObject:
    """
    One object a scene bundle places in its world.
    """

    name: str
    """
    The object's identifier in the bundle.
    """

    label: str
    """
    The name the object is shown by.
    """

    kind: ObjectKind = ObjectKind.OBJECT
    """
    What the object is of.
    """


@dataclass(frozen=True)
class SceneBundle:
    """
    A scene bundle held under cramera's scene roots.
    """

    name: str
    """
    The bundle's name.
    """

    directory: Path
    """
    The bundle's directory.
    """

    @classmethod
    def of(cls, requested: str | None) -> SceneBundle:
        """
        :param requested: The scene's name, or ``None`` for the default of the viewer's
            scene index.
        :return: The bundle.
        :raises NoSuchScene: When no bundle is held under the name, or none is the
            default.
        """
        name = requested or merged_scene_index().get("default")
        directory = (
            None if name is None else cramera_paths.resolve_scene_directory(name)
        )
        if directory is None:
            raise NoSuchScene(requested=requested)
        return cls(name=name, directory=directory)

    def objects(self) -> list[SceneObject]:
        """
        :return: The objects the bundle's scene places, each by its identifier.
        """
        scene = json.loads((self.directory / SceneFile.SCENE).read_text())
        return [
            SceneObject(name=entry["id"], label=entry["id"].replace("_", " ").title())
            for entry in scene.get("objects") or []
        ]
