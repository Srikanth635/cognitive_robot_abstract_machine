from __future__ import annotations

import itertools
from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy._typing import NDArray
from typing_extensions import Self, TypeVar, TYPE_CHECKING

from semantic_digital_twin.spatial_types import Point3, RotationMatrix, Quaternion

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import Body


def _best_xyz_permutation(axes: np.ndarray) -> np.ndarray:
    """
    Find a unique permutation of eigenvector columns that maximizes alignment
    with canonical axes x,y,z. This avoids duplicate-column picks.
    axes: (3,3) with eigenvectors in columns (np.linalg.eigh output).
    returns: (3,) permutation indices for columns → (x,y,z).
    """
    align = np.abs(axes.T @ np.eye(3))  # 3x3
    best_perm = None
    best_score = -np.inf
    for perm in itertools.permutations([0, 1, 2]):
        score = align[list(perm), range(3)].sum()
        if score > best_score:
            best_score = score
            best_perm = np.array(perm, dtype=int)
    return best_perm


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

    Off-diagonal elements of the inertia tensor. Note: these can be positive or negative.
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
    """

    @classmethod
    def from_values(cls, ixx: float, iyy: float, izz: float) -> Self:
        """Construct from scalar components (Ixx, Iyy, Izz)."""
        return cls(data=np.array([ixx, iyy, izz], dtype=float))


@dataclass(eq=False)
class PrincipalMoments(MomentsVector3):
    """
    Represents the three principal moments of inertia (I1, I2, I3) about the principal axes of the body.
    A principal moment is the eigenvalue of the inertia tensor corresponding to a principal axis.
    """

    @classmethod
    def from_values(cls, i1: float, i2: float, i3: float) -> Self:
        """Construct from scalar components (I1, I2, I3)."""
        return cls(data=np.array([i1, i2, i3], dtype=float))


def _project_to_so3(M: np.ndarray, *, atol: float = 1e-10) -> np.ndarray:
    """
    Project a near-rotation 3x3 matrix onto SO(3) using SVD:
        M = U S V^T  ->  R = U V^T   (and fix det sign if needed)
    """
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # Fix a reflection by flipping the last column of U
        U[:, -1] *= -1.0
        R = U @ Vt
    # Optional final clamp for tiny drift
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        R = (R + R.T) * 0.5  # symmetrize a hair; SVD above usually suffices
    return R


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

    @classmethod
    def from_quaternion(cls, x: float, y: float, z: float, w: float) -> Self:
        """
        Create PrincipalAxes from a quaternion that encodes the rotation
        **from canonical/world frame TO principal frame** (world→principal).

        Because PrincipalAxes stores principal→world, we invert it.
        """
        # Normalize defensively (in case upstream didn’t)
        q = np.array([x, y, z, w], dtype=float)
        n = np.linalg.norm(q)
        if n == 0:
            raise ValueError("Zero quaternion is invalid")
        q = q / n
        x, y, z, w = q

        # Build world→principal matrix from (x, y, z, w)
        # (standard Hamilton convention; adjust if your Quaternion uses a different one)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        R_wp = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

        # We need principal→world:
        R_pw = R_wp.T

        # Project to SO(3) to enforce orthonormality and det +1
        R_pw = _project_to_so3(R_pw)

        return cls(data=R_pw)

    def to_quaternion(self) -> Tuple[float, float, float, float]:
        """
        Convert PrincipalAxes (principal→world) to a unit quaternion (x, y, z, w)
        encoding **the same direction (principal→world)**.
        If you need world→principal, take the conjugate (x, y, z, w) -> (-x, -y, -z, w).
        """
        R = _project_to_so3(self.data)  # defensive; should already be SO(3)
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

        # Robust matrix→quat (x, y, z, w). This returns the principal→world quaternion.
        trace = m00 + m11 + m22
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s

        q = np.array([x, y, z, w], dtype=float)
        q /= np.linalg.norm(q)

        # Optional: pick a consistent hemisphere (e.g., w >= 0) to stabilize tests
        if q[3] < 0:
            q[:3] *= -1.0
            q[3] *= -1.0

        return float(q[0]), float(q[1]), float(q[2]), float(q[3])


@dataclass(eq=False)
class InertiaTensor(NPMatrix3x3):
    """
    Represents the inertia tensor of a body in a given coordinate frame.
    The inertia tensor is a symmetric positive semi-definite 3x3 matrix.
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
        moments_matrix = moments.as_matrix()
        I = axes @ moments_matrix @ axes.T()
        return cls(data=I.data)

    def to_principals(self) -> Tuple[PrincipalMoments, PrincipalAxes]:
        moments, axes = np.linalg.eigh(self.data)  # columns of vecs are eigenvectors

        # Find best permutation of axes to align with x,y,z
        perm = _best_xyz_permutation(axes)
        moments = moments[perm]
        axes = axes[:, perm]

        # After permutation, re-enforce right-handedness (permutation can flip parity)
        if np.linalg.det(axes) < 0:
            # Flip the column least aligned with its target axis to preserve alignment
            target = np.eye(3)
            dots = np.abs(np.sum(axes * target, axis=0))  # alignment with x,y,z
            flip_idx = int(np.argmin(dots))
            axes[:, flip_idx] *= -1.0

        return PrincipalMoments(data=moments), PrincipalAxes(data=axes)

    @classmethod
    def from_moments_products(
        cls, moments: MomentsOfInertia, products: ProductsOfInertia
    ):
        """
        Construct from moments and products of inertia.
        """
        moment_matrix = moments.as_matrix()
        products_matrix = products.as_matrix()
        return cls(data=moment_matrix.data + products_matrix.data)

    def to_inertia_values(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns the 6 unique values M(1,1), M(2,2), M(3,3), M(1,2), M(1,3), M(2,3) of the inertia tensor.
        """
        m11 = self.data[0, 0]
        m22 = self.data[1, 1]
        m33 = self.data[2, 2]
        m12 = self.data[0, 1]
        m13 = self.data[0, 2]
        m23 = self.data[1, 2]
        return m11, m22, m33, m12, m13, m23


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

    _reference_frame: Body = field(default=None, repr=False, compare=False)
    """
    The body to which this inertial property belongs.
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
