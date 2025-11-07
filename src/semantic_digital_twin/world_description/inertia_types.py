from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy._typing import NDArray
from typing_extensions import Self, TypeVar, TYPE_CHECKING

from semantic_digital_twin.spatial_types import Point3


@dataclass
class NPMatrix3x3:
    data: NDArray

    def __post_init__(self):
        assert self.data.shape == (3, 3), "Matrix must be 3x3"

    def __matmul__(self, other: GenericMatrix3x3Type) -> GenericMatrix3x3Type:
        return NPMatrix3x3(data=self.data @ other.data)


GenericMatrix3x3Type = TypeVar("GenericMatrix3x3Type", bound=NPMatrix3x3)


@dataclass
class NPVector3:
    data: NDArray

    def __post_init__(self):
        assert self.data.shape == (3,), "Vector must be 3-dimensional"

    @classmethod
    def from_values(cls, x: float, y: float, z: float) -> Self:
        return cls(data=np.array([x, y, z]))

    def to_values(self) -> Tuple[float, float, float]:
        """Return the tuple (x,y,z)"""
        return self.data[0], self.data[1], self.data[2]

    def as_matrix(self) -> NPMatrix3x3:
        return NPMatrix3x3(data=np.diag(self.data))


@dataclass(eq=False)
class ProductsOfInertia(NPVector3):
    """
    Represents the three products of inertia (Ixy, Ixz, Iyz)
    in the *current* coordinate frame.

    https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    """

    @classmethod
    def from_values(cls, ixy: float, ixz: float, iyz: float) -> Self:
        """Construct from scalar components (Ixy, Ixz, Iyz)."""
        return cls(data=np.array([ixy, ixz, iyz], dtype=float))

    def as_matrix(self) -> NPMatrix3x3:
        """
        Return a 3x3 matrix containing only the off-diagonal components:

            [[0, Ixy, Ixz],
             [Ixy, 0, Iyz],
             [Ixz, Iyz, 0]]
        """
        ixy, ixz, iyz = self.data
        mat = np.array([[0.0, ixy, ixz], [ixy, 0.0, iyz], [ixz, iyz, 0.0]], dtype=float)
        return NPMatrix3x3(data=mat)


@dataclass(eq=False)
class MomentsVector3(NPVector3, ABC):
    """
    Base class for moment vectors.
    """

    def __post_init__(self):
        assert np.all(self.data >= 0), "Moments must be non-negative"


@dataclass(eq=False)
class MomentsOfInertia(MomentsVector3):
    """
    Represents the three moments of inertia about the reference frame axes.
    These are the diagonal elements of the inertia tensor (Ixx, Iyy, Izz)
    in the *current* coordinate system.
    https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    """

    @classmethod
    def from_values(cls, ixx: float, iyy: float, izz: float) -> Self:
        """Construct from scalar components (Ixx, Iyy, Izz)."""
        ...


@dataclass(eq=False)
class PrincipalMoments(MomentsVector3):
    """
    Represents the three principal moments of inertia (I1, I2, I3) about the principal axes of the body.
    A principal moment is the eigenvalue of the inertia tensor corresponding to a principal axis.
    """

    @classmethod
    def from_values(cls, i1: float, i2: float, i3: float) -> Self:
        """Construct from scalar components (I1, I2, I3)."""
        ...


@dataclass
class PrincipalAxes(NPMatrix3x3):
    """
    The principal axes of the inertia tensor is a 3x3 matrix where each column is a principal axis.
    A principal axis is an eigenvector of the inertia tensor corresponding to a principal moment of inertia.
    """

    def __post_init__(self):
        super().__post_init__()
        assert np.allclose(self.data.T @ self.data, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(self.data), 1.0, atol=1e-10)

    def T(self) -> Self:
        return PrincipalAxes(data=self.data.T)


@dataclass(eq=False)
class InertiaTensor(NPMatrix3x3):
    """
    Represents the inertia tensor of a body in a given coordinate frame.
    The inertia tensor is a symmetric positive semi-definite 3x3 matrix.
    https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    """

    def __post_init__(self):
        super().__post_init__()
        assert np.allclose(self.data, self.data.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(self.data)
        assert np.all(eigvals >= -1e-12)
        assert np.isclose(np.trace(self.data), np.sum(eigvals), atol=1e-10)
        assert np.linalg.det(self.data) >= -1e-10

    @classmethod
    def from_principals(cls, axes: PrincipalAxes, moments: PrincipalMoments):
        """
        Construct from principal representation:
        R * diag(I1,I2,I3) * R^T
        """
        ...

    def to_principals(self) -> Tuple[PrincipalMoments, PrincipalAxes]:
        moments, axes = ...
        return PrincipalMoments(data=moments), PrincipalAxes(data=axes)

    @classmethod
    def from_moments_products(
        cls, moments: MomentsOfInertia, products: ProductsOfInertia
    ):
        """
        Construct from moments and products of inertia.
        Needed for urdf.
        """
        ...

    def to_moments_products(self) -> Tuple[MomentsOfInertia, ProductsOfInertia]:
        """
        Returns the 6 unique values M(1,1), M(2,2), M(3,3), M(1,2), M(1,3), M(2,3) of the inertia tensor.
        Needed for urdf.
        """
        moments, products = ...
        return moments, products

    @classmethod
    def from_principal_moment_quaternion(
        cls,
        principal_moments: PrincipalMoments,
        quaternion: Tuple[float, float, float, float],
    ) -> Self:
        """
        Construct from principal moments and quaternion.
        Needed for mujoco.
        """
        ...

    def to_principal_moment_quaternion(
        self,
    ) -> Tuple[PrincipalMoments, Tuple[float, float, float, float]]:
        """
        Returns the principal moments and quaternion of the inertia tensor.
        Needed for mujoco.
        """
        moment, quat = ...
        return moment, quat


@dataclass
class BodyInertial:
    """
    Represents the inertial properties of a body. https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial
    """

    mass: float = 1.0
    """
    The mass of the body in kilograms.
    """

    center_of_mass: Point3 = field(default_factory=Point3)
    """
    The center of mass of the body. If a force acts through the COM, the body experiences pure translation, no torque
    """

    inertia: InertiaTensor = field(default_factory=InertiaTensor)
    """
    The inertia tensor of the body about its center of mass, expressed in the body's local coordinate frame.
    """


@dataclass
class ConnectionInertial:
    stiffness: float = field(default=0.0)
    damping: float = field(default=0.0)
    armature: float = field(default=0.0)
    frictionloss: float = field(default=0.0)

    def __post_init__(self):
        assert self.stiffness >= 0, "Stiffness must be non-negative"
        assert self.damping >= 0, "Damping must be non-negative"
        assert self.armature >= 0, "Armature must be non-negative"
        assert self.frictionloss >= 0, "Friction loss must be non-negative"
