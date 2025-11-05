from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Optional, TYPE_CHECKING, Self, ClassVar

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.world_description.world_entity import (
        WorldEntity,
    )


@dataclass
class WorldEntityKwargsTracker:
    """
    Helps keep track of the world entities that have been parsed in a from_json call from SubclassJSONSerializer.
    """

    _data: Dict[PrefixedName, WorldEntity] = field(default_factory=dict)
    _world: Optional[World] = field(init=False)
    __world_entity_tracker: ClassVar[str] = "__world_entity_tracker"

    @classmethod
    def from_kwargs(cls, kwargs) -> Self:
        tracker = kwargs.get(cls.__world_entity_tracker) or cls()
        tracker._world = kwargs.get("world")
        kwargs[cls.__world_entity_tracker] = tracker
        return tracker

    def add_parsed_world_entity(self, world_entity: WorldEntity):
        self._data[world_entity.name] = world_entity

    def get_world_entity(self, name: PrefixedName) -> WorldEntity:
        world_entity = self._data.get(name)
        if world_entity is None:
            return self._world.get_kinematic_structure_entity_by_name(name)
        return world_entity
