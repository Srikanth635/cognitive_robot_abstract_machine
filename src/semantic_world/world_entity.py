from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Tuple

import numpy as np
from trimesh.proximity import closest_point, nearby_faces
from trimesh.sample import sample_surface_even, sample_surface
from scipy.stats import geom

from .geometry import Shape
from .prefixed_name import PrefixedName
from .spatial_types.spatial_types import TransformationMatrix, Expression
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

    def compute_closest_points_multi(self, others: list[Body], sample_size=25) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the closest points to each given body respectively.

        :param others: The list of bodies to compute the closest points to.
        :param sample_size: The number of samples to take from the surface of the other bodies.
        :return: A tuple containing: The points on the self body, the points on the other bodies, and the distances. All points are in the of this body.
        """
        query_points = []
        for other in others:
            # Calculate the closest vertex on this body to the other body
            closest_vert_id = self.collision[0].mesh.kdtree.query(self._world.compute_forward_kinematics_np(self, other)[:3, 3], k=1)[1]
            closest_vert = self.collision[0].mesh.vertices[closest_vert_id]

            # Compute the closest faces on the other body to the closes vertex
            faces = nearby_faces(other.collision[0].mesh, [self._world.compute_forward_kinematics_np(other, self)[:3, 3] + closest_vert])[0]
            face_weights = [0 for _ in range(len(other.collision[0].mesh.faces))]

            # Assign weights to the faces based on a geometric distribution
            for i, face_id in enumerate(faces):
                face_weights[face_id] = geom.pmf(i + 1, 0.5)  # Geometric distribution for face weights

            # Sample points on the surface of the other body
            q = sample_surface(other.collision[0].mesh, sample_size, face_weight=face_weights, seed=420)[0] + self._world.compute_forward_kinematics_np(self, other)[:3, 3]
            query_points.extend(q)

        # Actually compute the closest points
        points, dists = closest_point(self.collision[0].mesh, query_points)[:2]
        # Find the closest points for each body out of all the sampled points
        points = np.array(points).reshape(len(others), sample_size, 3)
        dists = np.array(dists).reshape(len(others), sample_size)
        dist_min = np.min(dists, axis=1)
        points_min_self = points[np.arange(len(others)), np.argmin(dists, axis=1), :]
        points_min_other = np.array(query_points).reshape(len(others), sample_size, 3)[np.arange(len(others)), np.argmin(dists, axis=1), :]
        return points_min_self, points_min_other, dist_min


class View(WorldEntity):
    """
    Represents a view on a set of bodies in the world.

    This class can hold references to certain bodies that gain meaning in this context.
    """


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
