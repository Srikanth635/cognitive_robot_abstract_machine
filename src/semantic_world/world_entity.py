from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Optional, TYPE_CHECKING, Set
import numpy as np
from numpy import ndarray

from .geometry import Shape, BoundingBox, BoundingBoxCollection
from .prefixed_name import PrefixedName
from .spatial_types.spatial_types import TransformationMatrix, Expression, Point3
from .spatial_types import spatial_types as cas
from .utils import IDGenerator

if TYPE_CHECKING:
    from .world import World

id_generator = IDGenerator()


@dataclass(unsafe_hash=True)
class WorldEntity:
    """
    A class representing an entity in the world.
    """

    _world: Optional[World] = field(default=None, repr=False, kw_only=True, hash=False)
    """
    The backreference to the world this entity belongs to.
    """

    _views: List[View] = field(default_factory=list, init=False, repr=False, hash=False)
    """
    The views this entity is part of.
    """


@dataclass
class Body(WorldEntity):
    """
    Represents a body in the world.
    A body is a semantic atom, meaning that it cannot be decomposed into meaningful smaller parts.
    """

    name: PrefixedName
    """
    The name of the link. Must be unique in the world.
    If not provided, a unique name will be generated.
    """

    visual: List[Shape] = field(default_factory=list, repr=False)
    """
    List of shapes that represent the visual appearance of the link.
    The poses of the shapes are relative to the link.
    """

    collision: List[Shape] = field(default_factory=list, repr=False)
    """
    List of shapes that represent the collision geometry of the link.
    The poses of the shapes are relative to the link.
    """

    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    def __post_init__(self):
        if not self.name:
            self.name = PrefixedName(f"body_{id_generator(self)}")

        if self._world is not None:
            self._world.kinematic_structure.add_body(self)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def has_collision(self) -> bool:
        return len(self.collision) > 0

    @property
    def bounding_box_collection(self) -> BoundingBoxCollection:
        """
        Return the bounding box collection of the link with the given name.
        This method computes the bounding box of the link in world coordinates by transforming the local axis-aligned
        bounding boxes of the link's geometry to world coordinates.

        :return: A BoundingBoxCollection containing the bounding boxes of the link's geometry in world
        """
        world = self._world
        body_transform: ndarray = world.compute_forward_kinematics_np(world.root, self)
        world_bboxes = []

        for shape in self.collision:
            shape_transform: ndarray = shape.origin.to_np()

            world_transform: ndarray = body_transform @ shape_transform
            body_pos = world_transform[:3, 3]
            body_rotation_matrix = world_transform[:3, :3]

            local_bb: BoundingBox = shape.as_bounding_box()

            # Get all 8 corners of the BB in link-local space
            corners = np.array([corner.to_np()[:3] for corner in local_bb.get_points()])  # shape (8, 3)

            # Transform each corner to world space: R * corner + T
            transformed_corners = (corners @ body_rotation_matrix.T) + body_pos

            # Compute world-space bounding box from transformed corners
            min_corner = np.min(transformed_corners, axis=0)
            max_corner = np.max(transformed_corners, axis=0)

            world_bb = BoundingBox.from_min_max(Point3.from_xyz(*min_corner), Point3.from_xyz(*max_corner))
            world_bboxes.append(world_bb)

        return BoundingBoxCollection(world_bboxes)


class View(WorldEntity):
    """
    Represents a view on a set of bodies in the world.

    This class can hold references to certain bodies that gain meaning in this context.
    """

    @property
    def aggregated_bodies(self) -> Set[Body]:
        """
        Recursively traverses the view and its attributes to find all bodies contained within it.

        :return: A set of bodies that are part of this view.
        """
        bodies: Set[Body] = set()
        visited: Set[int] = set()
        stack: deque = deque([self])

        while stack:
            obj = stack.pop()
            oid = id(obj)
            if oid in visited:
                continue
            visited.add(oid)

            if isinstance(obj, Body):
                bodies.add(obj)
                continue

            if isinstance(obj, View):
                stack.extend(obj.__dict__.values())
                continue

            if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
                stack.extend(obj)
        return bodies

    def as_bounding_box_collection(self) -> BoundingBoxCollection:
        """
        Returns a bounding box collection that contains the bounding boxes of all bodies in this view.
        """
        bbs = reduce(
            lambda accumulator, bb_collection: accumulator.merge(bb_collection),
            (body.bounding_box_collection for body in self.aggregated_bodies if body.has_collision())
        )
        return bbs

@dataclass
class Connection(WorldEntity):
    """
    Represents a connection between two bodies in the world.
    """

    parent: Body
    """
    The parent body of the connection.
    """

    child: Body
    """
    The child body of the connection.
    """

    origin: TransformationMatrix = None
    """
    The origin of the connection.
    """

    def __post_init__(self):
        if self.origin is None:
            self.origin = TransformationMatrix()
        self.origin.reference_frame = self.parent.name
        self.origin.child_frame = self.child.name

    def __hash__(self):
        return hash((self.parent, self.child))

    def __eq__(self, other):
        return self.name == other.name

    @property
    def name(self):
        return PrefixedName(f'{self.parent.name.name}_T_{self.child.name.name}', prefix=self.child.name.prefix)

    # @lru_cache(maxsize=None)
    def origin_as_position_quaternion(self) -> Expression:
        position = self.origin.to_position()[:3]
        orientation = self.origin.to_quaternion()
        return cas.vstack([position, orientation]).T
