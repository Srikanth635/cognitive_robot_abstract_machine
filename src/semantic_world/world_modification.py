from __future__ import annotations

import inspect
from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import List, TYPE_CHECKING, Dict, Any, Self, Callable, Optional, Type

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
    connection_type: Type[Connection]
    connection_kwargs: Dict[str, Any]
    parent_name: PrefixedName
    child_name: PrefixedName

    @classmethod
    def blacklisted_kwargs(cls):
        return ["parent", "child", "_world", "_views"]

    def __post_init__(self):
        self.connection_kwargs = {k: v for k, v in self.connection_kwargs.items() if k not in self.blacklisted_kwargs()}

    def __call__(self, world: World):
        parent = world.get_kinematic_structure_entity_by_name(self.parent_name)
        child = world.get_kinematic_structure_entity_by_name(self.child_name)
        connection = self.connection_type(parent=parent, child=child, **self.connection_kwargs, _world=world)
        world.add_connection(connection)

    @classmethod
    def from_connection(cls, connection: Connection,):
        parent_name = connection.parent.name
        child_name = connection.child.name
        connection_kwargs = {k: v for k, v in connection.__dict__.items() if k not in cls.blacklisted_kwargs()}
        return cls(connection_type=type(connection), connection_kwargs=connection_kwargs, parent_name=parent_name, child_name=child_name)

    def to_json(self):
        result = {**super().to_json(), "parent_name": self.parent_name.to_json(),
                "child_name": self.child_name.to_json()}

        init = inspect.signature(self.connection_type.__init__)
        # bind args and kwargs
        bound = init.bind_partial(
            self, **self.connection_kwargs
        )

        for k, v in bound.arguments.items():
            if k == "self":
                continue
            result[k] = v.to_json()

        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        parent_name = data["parent_name"]
        child = Body.from_json(data["child_name"])
        return cls(connection=Connection(parent=parent, child=child))

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

    skip_callbacks: Optional[List] = None

    def __call__(self, world: World):
        with world.modify_world(skip_callbacks=self.skip_callbacks):
            for modification in self.modifications:
                modification(world)
