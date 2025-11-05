from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.world_description.world_entity import (
        WorldEntity,
    )


@dataclass
class ParsedWorldEntities:
    data: Dict[PrefixedName, WorldEntity] = field(default_factory=dict)

    def add_parsed_world_entities(self, world_entities: List[WorldEntity]):
        for world_entity in world_entities:
            self.data[world_entity.name] = world_entity

    def get_world_entity(self, name: PrefixedName) -> Optional[WorldEntity]:
        return self.data.get(name)
