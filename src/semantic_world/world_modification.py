from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import List, TYPE_CHECKING, Dict, Any, Self

from random_events.utils import SubclassJSONSerializer

from .prefixed_name import PrefixedName
from .world_entity import Body, Connection

if TYPE_CHECKING:
    from .world import World


@dataclass
class WorldModelModification(SubclassJSONSerializer, ABC):
    """
    A record of a modification to the model (structure) of the world.
    This includes add/remove body and add/remove connection.
    """

    @abstractmethod
    def __call__(self, world: World):
        ...

@dataclass
class AddBodyModification(WorldModelModification):
    body: Body

    def __call__(self, world: World):
        world.add_kinematic_structure_entity(self.body)

    def to_json(self):
        return {**super().to_json(), "body": self.body.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(body=Body.from_json(data["body"]))

@dataclass
class RemoveBodyModification(WorldModelModification):
    body_name: PrefixedName

    def __call__(self, world: World):
        world.remove_kinematic_structure_entity(world.get_kinematic_structure_entity_by_name(self.body_name))

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "body_name": self.body_name.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(body_name=PrefixedName.from_json(data["body_name"]))

@dataclass
class AddConnectionModification(WorldModelModification):
    connection: Connection

    def __call__(self, world: World):
        world.add_connection(self.connection)

@dataclass
class RemoveConnectionModification(WorldModelModification):
    connection_name: PrefixedName

    def __call__(self, world: World):
        world.remove_connection(world.get_connection_by_name(self.connection_name))

@dataclass
class WorldModelModificationBlock:
    """
    A sequence of WorldModelModifications that were applied to the world within one @modifies_world call.
    """
    modifications: List[WorldModelModification]

    def __call__(self, world: World):
        with world.modify_world():
            for modification in self.modifications:
                modification(world)
