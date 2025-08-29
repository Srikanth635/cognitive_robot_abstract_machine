from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Dict, Any, Self, Optional, Callable, ClassVar

from random_events.utils import SubclassJSONSerializer, recursive_subclasses

from .connection_factories import ConnectionFactory
from .degree_of_freedom import DegreeOfFreedom
from .prefixed_name import PrefixedName
from .world import World, FunctionStack
from .world_entity import Body


@dataclass
class WorldModelModification(SubclassJSONSerializer, ABC):
    """
    A record of a modification to the model (structure) of the world.
    This includes add/remove body and add/remove connection.
    """

    name: ClassVar[Optional[str]] = None

    @abstractmethod
    def __call__(self, world: World): ...

    @classmethod
    @abstractmethod
    def _from_kwargs(cls, kwargs: Dict[str, Any]):
        raise NotImplementedError

    @classmethod
    def from_world_modification(cls, call: Callable, kwargs: Dict[str, Any]):
        for subclass in recursive_subclasses(cls):
            if subclass.name == call.__name__:
                return subclass._from_kwargs(kwargs)
        raise ValueError(f"Unknown world modification: {call.__name__}")


@dataclass
class AddBodyModification(WorldModelModification):
    body: Body
    name = World.add_kinematic_structure_entity.__name__

    @classmethod
    def _from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"])

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
    name = World.remove_kinematic_structure_entity.__name__

    @classmethod
    def _from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"].name)

    def __call__(self, world: World):
        world.remove_kinematic_structure_entity(
            world.get_kinematic_structure_entity_by_name(self.body_name)
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "body_name": self.body_name.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(body_name=PrefixedName.from_json(data["body_name"]))


@dataclass
class AddConnectionModification(WorldModelModification):
    connection_factory: ConnectionFactory
    name = World.add_connection.__name__

    @classmethod
    def _from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(ConnectionFactory.from_connection(kwargs["connection"]))

    @classmethod
    def blacklisted_kwargs(cls):
        return ["parent", "child", "_world", "_views"]

    def __call__(self, world: World):
        connection = self.connection_factory.create(world)
        world.add_connection(connection)

    def to_json(self):
        return {
            **super().to_json(),
            "connection_factory": self.connection_factory.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(
            connection_factory=ConnectionFactory.from_json(data["connection_factory"])
        )


@dataclass
class RemoveConnectionModification(WorldModelModification):
    connection_name: PrefixedName
    name = World.remove_connection.__name__

    @classmethod
    def _from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["connection"].name)

    def __call__(self, world: World):
        world.remove_connection(world.get_connection_by_name(self.connection_name))

    def to_json(self):
        return {
            **super().to_json(),
            "connection_name": self.connection_name.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(connection_name=PrefixedName.from_json(data["connection_name"]))


@dataclass
class AddDegreeOfFreedomModification(WorldModelModification):
    dof: DegreeOfFreedom

    name = World.add_degree_of_freedom.__name__

    @classmethod
    def _from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(dof=kwargs["dof"])

    def __call__(self, world: World):
        world.add_degree_of_freedom(self.dof)

    def to_json(self):
        return {
            **super().to_json(),
            "dof": self.dof.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(dof=DegreeOfFreedom.from_json(data["dof"]))



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

    @classmethod
    def from_modifications(cls, modifications: FunctionStack):
        return cls(
            [
                WorldModelModification.from_world_modification(call, kwargs)
                for call, kwargs in modifications
            ]
        )
