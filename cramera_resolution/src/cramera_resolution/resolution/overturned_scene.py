"""
A variant of a scene bundle in which one object is turned upside down where it stands:
for testing whether a resolver reads that the object's opening faces down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Dict, List

from cramera_resolution.resolution.scene_world import SCENE_FILE, SceneKey, SceneWorldBuilder

HALF_TURN_ABOUT_X = [1.0, 0.0, 0.0, 0.0]
"""
The orientation quaternion (x, y, z, w) of a half turn about the world's x axis, which
turns an upright object upside down.
"""

SPAWN_POSITION_LENGTH = 3
"""
How many values of a spawn pose give the position; the rest is the orientation.
"""

SPAWN_HEIGHT_INDEX = 2
"""
Where the height is among the values of a spawn pose.
"""


@dataclass
class ObjectNotInScene(LookupError):
    """
    The scene has no object with the identifier to turn over.
    """

    identifier: str
    """
    The identifier asked for.
    """

    def __post_init__(self):
        super().__init__(f"the scene has no object {self.identifier!r}")


@dataclass
class OverturnedScene:
    """
    A scene bundle with one object turned upside down, resting on the surface it stood
    on.
    """

    bundle: Path
    """
    The bundle of the scene the variant is made from; left as it is.
    """

    identifier: str
    """
    The identifier of the object to turn over.
    """

    @classmethod
    def of_body(cls, bundle: Path, body_name: str) -> OverturnedScene:
        """
        :param bundle: The scene's bundle.
        :param body_name: The name of the body of the object to turn over.
        :return: The variant turning that object over.
        :raises ObjectNotInScene: When no object of the scene has that body.
        """
        scene = json.loads((bundle / SCENE_FILE).read_text())
        for entry in scene.get(SceneKey.OBJECTS) or []:
            if Path(entry[SceneKey.MESH]).name == body_name:
                return cls(bundle=bundle, identifier=entry[SceneKey.IDENTIFIER])
        raise ObjectNotInScene(identifier=body_name)

    @property
    def body_name(self) -> str:
        """
        :return: The name of the body the object has in the scene's world.
        """
        scene = json.loads((self.bundle / SCENE_FILE).read_text())
        return Path(self.entry_in(scene)[SceneKey.MESH]).name

    def write_to(self, destination: Path) -> Path:
        """
        Write the variant as a bundle that shares every file of the original but the
        scene's description.

        :param destination: A directory that does not exist yet.
        :return: The directory of the new bundle.
        :raises ObjectNotInScene: When the scene has no such object.
        """
        scene = json.loads((self.bundle / SCENE_FILE).read_text())
        entry = self.entry_in(scene)
        entry[SceneKey.SPAWN] = self.overturned_spawn(entry)
        destination.mkdir(parents=True)
        for path in self.bundle.iterdir():
            if path.name != SCENE_FILE:
                (destination / path.name).symlink_to(path.resolve())
        (destination / SCENE_FILE).write_text(json.dumps(scene, indent=1))
        return destination

    def entry_in(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        :return: The scene's description of the object.
        :raises ObjectNotInScene: When the scene has no such object.
        """
        matching: List[Dict[str, Any]] = [
            entry
            for entry in scene.get(SceneKey.OBJECTS) or []
            if entry[SceneKey.IDENTIFIER] == self.identifier
        ]
        if not matching:
            raise ObjectNotInScene(identifier=self.identifier)
        return matching[0]

    def overturned_spawn(self, entry: Dict[str, Any]) -> List[float]:
        """
        :return: The object's spawn pose turned upside down, raised by the height of its
            box so that its lowest point stays where it was.
        """
        position = list(entry[SceneKey.SPAWN][:SPAWN_POSITION_LENGTH])
        world = SceneWorldBuilder().build(self.bundle)
        body = world.get_body_by_name(Path(entry[SceneKey.MESH]).name)
        box = body.visual.as_bounding_box_collection_in_frame(world.root).bounding_box()
        position[SPAWN_HEIGHT_INDEX] = (
            box.min_z + box.max_z - position[SPAWN_HEIGHT_INDEX]
        )
        return [*position, *HALF_TURN_ABOUT_X]
