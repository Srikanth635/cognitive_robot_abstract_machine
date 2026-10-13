"""
A resolved slot's value as data a run keeps, and that data read back as a value in a
world: a world object by its body's name, an enumeration member by its name, any other
value as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm_reasoner.resolver.reflection import union_alternatives
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Any


@dataclass
class NoObjectInWorld(LookupError):
    """
    The world holds no object of the kind the data names.
    """

    kind: type
    """
    The kind of object asked for.
    """

    name: str
    """
    The name of the object's body.
    """

    def __post_init__(self):
        super().__init__(f"the world holds no {self.kind.__name__} named {self.name}")


class SlotData:
    """
    Writes a slot's value as data and reads the data back as a value in a world.
    """

    @staticmethod
    def from_value(value: Any) -> Any:
        """
        :return: The value as data: a body by its name, an annotated object by its
            body's name, an enumeration member by its name, any other value as it is.
        """
        if isinstance(value, Body):
            return value.name.name
        if isinstance(value, HasRootBody):
            return value.root.name.name
        if isinstance(value, Enum):
            return value.name
        return value

    @staticmethod
    def to_value(world: World, kind: Any, data: Any) -> Any:
        """
        :param world: The world the value is found in.
        :param kind: The type the slot declares.
        :param data: The slot's value as data.
        :return: The value: a body by its name, an annotated object by its body's name,
            an enumeration member by its name, nothing for an optional slot left empty,
            any other value as it is.
        :raises NoObjectInWorld: When the world holds no such object.
        """
        if data is None:
            return None
        alternatives = union_alternatives(kind)
        if alternatives is not None and len(alternatives) == 1:
            return SlotData.to_value(world, alternatives[0], data)
        if not isinstance(kind, type):
            return data
        if issubclass(kind, Body):
            return world.get_body_by_name(data)
        if issubclass(kind, Enum):
            return kind[data]
        if not issubclass(kind, HasRootBody):
            return data
        found = [
            annotation
            for annotation in world.get_semantic_annotations_by_type(kind)
            if annotation.root.name.name == data
        ]
        if not found:
            raise NoObjectInWorld(kind=kind, name=data)
        return found[0]
