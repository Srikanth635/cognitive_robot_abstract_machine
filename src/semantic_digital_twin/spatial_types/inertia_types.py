from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from typing import Optional

import casadi as ca
from typing_extensions import Self

from semantic_digital_twin.exceptions import WrongDimensionsError
from semantic_digital_twin.spatial_types import Point3, Vector3, RotationMatrix
from semantic_digital_twin.spatial_types.spatial_types import (
    SymbolicType,
    ReferenceFrameMixin,
    MatrixOperationsMixin,
    Matrix2dData,
    Expression,
    ScalarData,
)


@dataclass
class Inertial:
    """
    Represents the inertial properties of a body. https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial
    """

    mass: float = 1.0
    """
    The mass of the body. 
    """

    center_of_mass: Point3 = field(default_factory=Point3)
    """
    The center of mass of the body. If a force acts through the COM, the body experiences pure translation, no torque
    """

    diagonal_inertia: Vector3 = field(default_factory=Vector3)
    """
    The diagonal elements of the inertia matrix.
    """
    principal_axes: RotationMatrix = field(default_factory=RotationMatrix)

    def __post_init__(self):
        assert self.mass >= 0, "Mass must be non negative"


@dataclass
class Matrix3x3(SymbolicType, ReferenceFrameMixin, MatrixOperationsMixin):
    """
    Class to represent a 4x4 symbolic rotation matrix tied to kinematic references.

    This class provides methods for creating and manipulating rotation matrices within the context
    of kinematic structures. It supports initialization using data such as quaternions, axis-angle,
    other matrices, or directly through vector definitions. The primary purpose is to facilitate
    rotational transformations and computations in a symbolic context, particularly for applications
    like robotic kinematics or mechanical engineering.
    """

    data: InitVar[Optional[Matrix2dData]] = None
    """
    A 4x4 matrix of some form that represents the rotation matrix.
    """

    sanity_check: InitVar[bool] = field(kw_only=True, default=True)
    """
    Whether to perform a sanity check on the matrix data. Can be skipped for performance reasons.
    """

    casadi_sx: ca.SX = field(kw_only=True, default_factory=lambda: ca.SX.eye(3))

    def __post_init__(self, data: Optional[Matrix2dData], sanity_check: bool):
        if data is None:
            return
        self.casadi_sx = Expression(data=data).casadi_sx
        if sanity_check:
            self._validate()

    def _validate(self):
        if self.shape[0] != 3 or self.shape[1] != 3:
            raise WrongDimensionsError(
                expected_dimensions=(3, 3), actual_dimensions=self.shape
            )

    def __matmul__(self, other: Matrix3x3) -> Matrix3x3:
        return Matrix3x3(casadi_sx=self.casadi_sx @ other.casadi_sx)


@dataclass(eq=False)
class PrincipalMoments(SymbolicType, ReferenceFrameMixin, MatrixOperationsMixin):
    i_1: InitVar[ScalarData]

    i_2: InitVar[ScalarData]

    i_3: InitVar[ScalarData]

    casadi_sx: ca.SX = field(
        kw_only=True, default_factory=lambda: ca.SX([0.0, 0.0, 0.0])
    )

    def __post_init__(self, i_1, i_2, i_3):
        if min(i_1, i_2, i_3) <= 0:
            raise ValueError("Principal moments must be positive.")
        self[0] = i_1
        self[1] = i_2
        self[2] = i_3

    def as_matrix(self):
        matrix_3x3 = [
            [self[0], 0, 0],
            [0, self[1], 0],
            [0, 0, self[2]],
        ]
        return Matrix3x3(data=matrix_3x3)


@dataclass
class PrincipalAxes(Matrix3x3):

    def T(self) -> Self:
        return PrincipalAxes(casadi_sx=self.casadi_sx.T)

    def __matmul__(self, other):
        if isinstance(other, PrincipalAxes):
            return PrincipalAxes(casadi_sx=self.casadi_sx @ other.casadi_sx)
        return super().__matmul__(other)


@dataclass(eq=False)
class FullInertiaMatrix(SymbolicType, ReferenceFrameMixin, MatrixOperationsMixin):
    """
    Class to represent a 4x4 symbolic rotation matrix tied to kinematic references.

    This class provides methods for creating and manipulating rotation matrices within the context
    of kinematic structures. It supports initialization using data such as quaternions, axis-angle,
    other matrices, or directly through vector definitions. The primary purpose is to facilitate
    rotational transformations and computations in a symbolic context, particularly for applications
    like robotic kinematics or mechanical engineering.
    """

    data: InitVar[Optional[Matrix3x3]] = None
    """
    A 4x4 matrix of some form that represents the rotation matrix.
    """

    sanity_check: InitVar[bool] = field(kw_only=True, default=True)
    """
    Whether to perform a sanity check on the matrix data. Can be skipped for performance reasons.
    """

    casadi_sx: ca.SX = field(kw_only=True, default_factory=lambda: ca.SX.eye(3))

    def __post_init__(self, data: Optional[Matrix2dData], sanity_check: bool):
        if data is None:
            return
        self.casadi_sx = Expression(data=data).casadi_sx
        if sanity_check:
            self._validate()

    def _validate(self):
        if self.shape[0] != 3 or self.shape[1] != 3:
            raise WrongDimensionsError(
                expected_dimensions=(3, 3), actual_dimensions=self.shape
            )

    @classmethod
    def from_principal(cls, axes: PrincipalAxes, moments: PrincipalMoments):
        """
        Construct from principal representation:
        R * diag(I1,I2,I3) * R^T
        """
        moments_matrix = moments.as_matrix()
        I = axes @ moments_matrix @ axes.T()
        return cls(casadi_sx=I.casadi_sx)
