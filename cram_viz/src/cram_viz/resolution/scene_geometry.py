"""
The geometry of a scene's world as a fact the resolver reads: where every annotated
object stands, the box it fills and the centre of its top (a container's opening), and
where each robot's base stands and faces, in the world frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from krrood.entity_query_language.query.match import Match
from llm_reasoner.resolver.context import PromptContextProvider
from semantic_digital_twin.robots.robot_parts import AbstractRobot, AbstractRobotPart
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any, Dict, List, Optional

COORDINATE_DIGITS = 3
"""
The decimals a coordinate is stated with: millimetres.
"""

ROBOT_KINDS = (AbstractRobot, AbstractRobotPart)
"""
The annotations of a robot and of its parts, which are not objects of the scene: the
robot is stated by where its base stands.
"""

GEOMETRY_HEADING = (
    "The geometry of the scene: every object's origin, the box it fills and the centre "
    "of its top, and where each robot base stands and faces; metres and radians, in the "
    "world frame."
)
"""
The sentence introducing the geometry.
"""


class GeometryKey(StrEnum):
    """
    The keys the geometry is stated with.
    """

    OBJECTS = "objects"
    ROBOT_BASES = "robot_bases"
    NAME = "name"
    KIND = "kind"
    ORIGIN = "origin"
    BOUNDING_BOX = "bounding_box"
    MINIMUM = "min"
    MAXIMUM = "max"
    TOP_CENTRE = "top_centre"
    POSITION = "position"
    YAW = "yaw"


def rounded(values: List[float]) -> List[float]:
    """
    :return: The coordinates rounded to :data:`COORDINATE_DIGITS` decimals.
    """
    return [round(float(value), COORDINATE_DIGITS) for value in values]


@dataclass
class SceneGeometry(PromptContextProvider):
    """
    States the geometry of a scene's world in the situation section of the resolver's
    prompt.
    """

    world: World
    """
    The scene's world.
    """

    def context_for(self, expression: Optional[Match]) -> str:
        return f"{GEOMETRY_HEADING}\n{self.as_json_text()}"

    def as_json_text(self) -> str:
        """
        :return: The geometry as JSON text.
        """
        return json.dumps(
            {
                GeometryKey.OBJECTS: [
                    self.object_geometry(annotation)
                    for annotation in self.world.get_semantic_annotations_by_type(
                        HasRootBody
                    )
                    if not isinstance(annotation, ROBOT_KINDS)
                ],
                GeometryKey.ROBOT_BASES: [
                    self.base_geometry(connection.child)
                    for connection in self.world.connections
                    if isinstance(connection, OmniDrive)
                ],
            },
            indent=2,
        )

    def object_geometry(self, annotation: HasRootBody) -> Dict[str, Any]:
        """
        :return: An annotated object's kind, body name, origin, world-frame bounding box
            and the centre of the box's top.
        """
        body = annotation.root
        box = body.visual.as_bounding_box_collection_in_frame(
            self.world.root
        ).bounding_box()
        return {
            GeometryKey.KIND: type(annotation).__name__,
            GeometryKey.NAME: body.name.name,
            GeometryKey.ORIGIN: rounded(body.global_pose.to_position().to_np()[:3]),
            GeometryKey.BOUNDING_BOX: {
                GeometryKey.MINIMUM: rounded([box.min_x, box.min_y, box.min_z]),
                GeometryKey.MAXIMUM: rounded([box.max_x, box.max_y, box.max_z]),
            },
            GeometryKey.TOP_CENTRE: rounded(
                [
                    0.5 * (box.min_x + box.max_x),
                    0.5 * (box.min_y + box.max_y),
                    box.max_z,
                ]
            ),
        }

    @staticmethod
    def base_geometry(base: Body) -> Dict[str, Any]:
        """
        :return: A robot base's name, position on the floor and heading.
        """
        pose = base.global_pose
        return {
            GeometryKey.NAME: base.name.name,
            GeometryKey.POSITION: rounded(pose.to_position().to_np()[:2]),
            GeometryKey.YAW: round(
                float(pose.to_quaternion().to_rpy()[2]), COORDINATE_DIGITS
            ),
        }
