from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ormatic.dao import AlternativeMapping, T

from ..prefixed_name import PrefixedName
from ..spatial_types import RotationMatrix, Vector3, Point3, TransformationMatrix
from ..spatial_types.spatial_types import Quaternion
from ..spatial_types.symbol_manager import symbol_manager
from ..world import World, Body
from ..world_entity import Connection, View


@dataclass
class WorldMapping(AlternativeMapping[World]):
    bodies: List[Body]
    connections: List[Connection]
    views: List[View]

    @classmethod
    def create_instance(cls, obj: World):
        return cls(obj.bodies, obj.connections, obj.views)

    def create_from_dao(self) -> World:
        result = World()

        with result.modify_world():
            for body in self.bodies:
                result.add_body(body)
            for connection in self.connections:
                result.add_connection(connection)
            for view in self.views:
                result.add_view(view)

        return result


@dataclass
class Vector3Mapping(AlternativeMapping[Vector3]):
    reference_frame: PrefixedName

    x: float
    y: float
    z: float

    @classmethod
    def create_instance(cls, obj: Vector3):
        x, y, z, _ = symbol_manager.evaluate_expr(obj).tolist()
        return cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)

    def create_from_dao(self) -> Vector3:
        return Vector3.from_xyz(x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame)


@dataclass
class Point3Mapping(AlternativeMapping[Point3]):
    reference_frame: PrefixedName

    x: float
    y: float
    z: float

    @classmethod
    def create_instance(cls, obj: Point3):
        x, y, z, _ = symbol_manager.evaluate_expr(obj).tolist()
        result = cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)
        return result

    def create_from_dao(self) -> Point3:
        return Point3.from_xyz(x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame)

@dataclass
class QuaternionMapping(AlternativeMapping[Quaternion]):
    reference_frame: PrefixedName
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def create_instance(cls, obj: Quaternion):
        x, y, z, w = symbol_manager.evaluate_expr(obj).tolist()
        result = cls(x=x, y=y, z=z, w=w, reference_frame=obj.reference_frame)
        return result

    def create_from_dao(self) -> Quaternion:
        return Quaternion.from_xyzw(x=self.x, y=self.y, z=self.z, w=self.w, reference_frame=self.reference_frame)

@dataclass
class RotationMatrixMapping(AlternativeMapping[RotationMatrix]):
    reference_frame: PrefixedName
    rotation: Quaternion

    @classmethod
    def create_instance(cls, obj: RotationMatrix):
        return cls(reference_frame=obj.reference_frame, rotation=obj.to_quaternion())

    def create_from_dao(self) -> RotationMatrix:
        return RotationMatrix.from_quaternion(self.rotation)

@dataclass
class TransformationMatrixMapping(AlternativeMapping[TransformationMatrix]):
    reference_frame: PrefixedName
    child_frame: Optional[PrefixedName]
    position: Point3
    rotation: Quaternion

    @classmethod
    def create_instance(cls, obj: TransformationMatrix):
        position = obj.to_position()
        rotation = obj.to_quaternion()
        result = cls(reference_frame=obj.reference_frame, child_frame=obj.child_frame, position=position,
                     rotation=rotation)
        return result

    def create_from_dao(self) -> TransformationMatrix:
        return TransformationMatrix.from_point_rotation_matrix(
            point=self.position,
            rotation_matrix=RotationMatrix.from_quaternion(self.rotation), reference_frame=self.reference_frame,
            child_frame=self.child_frame,)
