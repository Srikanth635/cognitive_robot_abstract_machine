from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Optional, TYPE_CHECKING, Self, ClassVar, Any

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.world_description.world_entity import (
        WorldEntity,
    )


@dataclass
class WorldEntityKwargsTracker:
    """
    Keep track of the world entities that have been parsed in a from_json call from SubclassJSONSerializer.
    Initialize this class on the top level with from_kwargs(kwargs), then pass **kwargs to from_json.
    A kinematic structure entity will automatically add itself to the tracker if it is created during parsing, making
    it available objects that are from_json'ed later.

    """

    _data: Dict[PrefixedName, WorldEntity] = field(default_factory=dict)
    _world: Optional[World] = field(init=False, default=None)
    __world_entity_tracker: ClassVar[str] = "__world_entity_tracker"

    @classmethod
    def from_kwargs(cls, kwargs) -> Self:
        """
        Retrieve the tracker from the kwargs, or initialize a new one if it doesn't exist.
        :param kwargs: **kwargs from from_json call.
        """
        tracker = kwargs.get(cls.__world_entity_tracker) or cls()
        kwargs[cls.__world_entity_tracker] = tracker
        return tracker

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Create a new tracker from a world.
        :param world: A world instance that will be used as a backup to look for world entities.
        """
        tracker = cls()
        tracker._world = world
        return tracker

    def create_kwargs(self) -> Dict[str, Self]:
        return {self.__world_entity_tracker: self}

    def add_to_kwargs(self, kwargs: Dict[str, Any]):
        kwargs[self.__world_entity_tracker] = self

    def add_parsed_world_entity(self, world_entity: WorldEntity):
        self._data[world_entity.name] = world_entity

    def get_world_entity(self, name: PrefixedName) -> WorldEntity:
        world_entity = self._data.get(name)
        if world_entity is None and self._world is not None:
            return self._world.get_kinematic_structure_entity_by_name(name)
        return world_entity
