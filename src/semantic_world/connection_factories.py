from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, Dict, Any, TypeVar, Generic

from ormatic.dao import HasGeneric
from random_events.utils import SubclassJSONSerializer, recursive_subclasses

from .connections import FixedConnection, PrismaticConnection, RevoluteConnection, Connection6DoF
from .degree_of_freedom import DegreeOfFreedom
from .prefixed_name import PrefixedName
from .spatial_types.symbol_manager import symbol_manager
from .world import World
from .world_entity import Connection
from . import spatial_types as cas

T = TypeVar('T')

@dataclass
class ConnectionFactory(HasGeneric[T], SubclassJSONSerializer, ABC):
    """
    Factory for creating connections.
    This class can be used to serialize connections indirectly.
    """

    name: PrefixedName
    parent_name: PrefixedName
    child_name: PrefixedName

    @classmethod
    def from_connection(cls, connection: Connection) -> Self:
        for factory in recursive_subclasses(cls):
            if factory.original_class() == connection.__class__:
                return factory._from_connection(connection)
        raise ValueError(f"Unknown connection type: {connection.name}")

    @classmethod
    @abstractmethod
    def _from_connection(cls, connection: Connection) -> Self:
        """
        Create a connection factory from a connection.

        :param connection: The connection to create the factory from.
        :return: The created connection factory.
        """
        raise NotImplementedError

    @abstractmethod
    def create(self, world: World) -> T:
        """
        Create the connection in a given world.

        :param world: The world in which to create the connection.
        :return: The created connection.
        """
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "name": self.name.to_json(),
            "parent_name": self.parent_name.to_json(),
            "child_name": self.child_name.to_json(),
        }

@dataclass
class FixedConnectionFactory(ConnectionFactory[FixedConnection]):

    @classmethod
    def _from_connection(cls, connection: Connection) -> Self:
        return cls(name=connection.name,
                   parent_name=connection.parent.name,
                   child_name=connection.child.name,
                   )

    def create(self, world: World) -> Connection:
        parent = world.get_kinematic_structure_entity_by_name(self.parent_name)
        child = world.get_kinematic_structure_entity_by_name(self.child_name)
        return self.original_class()(parent=parent, child=child, name=self.name)

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=PrefixedName.from_json(data["name"]),
                   parent_name=PrefixedName.from_json(data["parent_name"]),
                   child_name=PrefixedName.from_json(data["child_name"]),
                   )

@dataclass
class PrismaticConnectionFactory(ConnectionFactory[PrismaticConnection]):
    axis: cas.Vector3
    multiplier: float
    offset: float
    dof: DegreeOfFreedom

    @classmethod
    def _from_connection(cls, connection: PrismaticConnection) -> Self:
        return cls(name=connection.name,
                   parent_name=connection.parent.name,
                   child_name=connection.child.name,
                   axis=connection.axis,
                   multiplier=connection.multiplier,
                   offset=connection.offset,
                   dof=connection.dof,
                   )

    def create(self, world: World) -> Connection:
        parent = world.get_kinematic_structure_entity_by_name(self.parent_name)
        child = world.get_kinematic_structure_entity_by_name(self.child_name)
        return self.original_class()(parent=parent, child=child, name=self.name, axis=self.axis, multiplier=self.multiplier, offset=self.offset, dof=self.dof)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name.to_json(),
            "parent_name": self.parent_name.to_json(),
            "child_name": self.child_name.to_json(),
            "axis": symbol_manager.evaluate_expr(self.axis).tolist(),
            "multiplier": self.multiplier,
            "offset": self.offset,
            "dof": self.dof.to_json()
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=PrefixedName.from_json(data["name"]),
                   parent_name=PrefixedName.from_json(data["parent_name"]),
                   child_name=PrefixedName.from_json(data["child_name"]),
                   axis=cas.Vector3.from_iterable(data["axis"]),
                   multiplier=data["multiplier"],
                   offset=data["offset"],
                   dof=DegreeOfFreedom.from_json(data["dof"]),
                   )

@classmethod
class RevoluteConnectionFactory(ConnectionFactory[RevoluteConnection]):
    axis: cas.Vector3
    multiplier: float
    offset: float
    dof: DegreeOfFreedom

    @classmethod
    def _from_connection(cls, connection: RevoluteConnection) -> Self:
        return cls(name=connection.name,
                   parent_name=connection.parent.name,
                   child_name=connection.child.name,
                   axis=connection.axis,
                   multiplier=connection.multiplier,
                   offset=connection.offset,
                   dof=connection.dof,
                   )

    def create(self, world: World) -> T:
        parent = world.get_kinematic_structure_entity_by_name(self.parent_name)
        child = world.get_kinematic_structure_entity_by_name(self.child_name)
        return self.original_class()(parent=parent, child=child, name=self.name, axis=self.axis, multiplier=self.multiplier, offset=self.offset, dof=self.dof)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name.to_json(),
            "parent_name": self.parent_name.to_json(),
            "child_name": self.child_name.to_json(),
            "axis": symbol_manager.evaluate_expr(self.axis).tolist(),
            "multiplier": self.multiplier,
            "offset": self.offset,
            "dof": self.dof.to_json()
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=PrefixedName.from_json(data["name"]),
                   parent_name=PrefixedName.from_json(data["parent_name"]),
                   child_name=PrefixedName.from_json(data["child_name"]),
                   axis=cas.Vector3.from_iterable(data["axis"]),
                   multiplier=data["multiplier"],
                   offset=data["offset"],
                   dof=DegreeOfFreedom.from_json(data["dof"]),
                   )

@dataclass
class Connection6DoFFactory(ConnectionFactory[Connection6DoF]):
    ...