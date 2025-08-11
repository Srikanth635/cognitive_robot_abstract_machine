import os
from dataclasses import dataclass

import trimesh

from ..prefixed_name import PrefixedName
from ..spatial_types.spatial_types import TransformationMatrix
from ..world import World
from ..world_entity import Body
from ..geometry import Mesh


@dataclass
class STLParser:
    """
    Adapter for STL files.
    """
    file_path: str
    """
    The path to the STL file.
    """

    def parse(self) -> World:
        """
        Parse the STL file to a body and return a world containing that body.

        :return: A World object containing the parsed body.
        """
        file_name = os.path.basename(self.file_path)

        mesh_shape = Mesh(origin=TransformationMatrix(), filename=self.file_path)
        body = Body(name=PrefixedName(file_name), collision=[mesh_shape], visual=[mesh_shape])

        world = World()
        world.add_body(body)

        return world
