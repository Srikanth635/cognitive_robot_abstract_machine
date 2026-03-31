from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from krrood.adapters.json_serializer import SubclassJSONSerializer


@dataclass
class Sage10kPhysicallyBasedRendering:
    metallic: float
    roughness: float


@dataclass
class Sage10kWall:
    id: str
    start_point: Sage10kPosition
    end_point: Sage10kPosition
    material: str
    height: float
    thickness: float
    material: str


@dataclass
class Sage10kObject:
    id: str
    room_id: str
    type: str
    description: str
    source: str
    source_id: str
    place_id: str
    place_guidance: str
    mass: float

    position: Sage10kPosition
    rotation: Sage10kRotation
    size: Sage10kSize
    physically_based_rendering: Sage10kPhysicallyBasedRendering


@dataclass
class Sage10kRotation:
    x: float
    y: float
    z: float


@dataclass
class Sage10kPosition:
    x: float
    y: float
    z: float


@dataclass
class Sage10kSize:
    height: float
    length: float
    width: float


@dataclass
class Sage10kRoom:
    id: str
    room_type: str
    dimensions: Sage10kSize
    position: Sage10kPosition
    floor_material: str
    objects: List[Sage10kObject] = field(default_factory=list)
    walls: List[Sage10kWall] = field(default_factory=list)


@dataclass
class Sage10kDoor:
    id: str
    wall_id: str
    position_on_wall: float
    width: float
    height: float
    door_type: str
    opens_inward: bool
    opening: bool
    door_material: str


@dataclass
class Sage10kScene(SubclassJSONSerializer):
    id: str
    building_style: str
    description: str
    created_from_text: str
    total_area: float
    rooms: List[Sage10kRoom] = field(default_factory=list)
