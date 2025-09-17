from enum import IntEnum
from typing import Union, Iterable, TypeVar

import numpy as np

from .spatial_types import (
    Point3,
    Vector3,
    TransformationMatrix,
    RotationMatrix,
    Quaternion,
)
from .symbolic_core import Symbol, Expression

NumericalScalar = Union[int, float, IntEnum]
NumericalArray = Union[np.ndarray, Iterable[NumericalScalar]]
Numerical2dMatrix = Union[np.ndarray, Iterable[NumericalArray]]
NumericalData = Union[NumericalScalar, NumericalArray, Numerical2dMatrix]

SymbolicScalar = Union[Symbol, Expression]
SymbolicArray = Union[Expression, Point3, Vector3, Quaternion]
Symbolic2dMatrix = Union[Expression, RotationMatrix, TransformationMatrix]
SymbolicData = Union[SymbolicScalar, SymbolicArray, Symbolic2dMatrix]

ScalarData = Union[NumericalScalar, SymbolicScalar]
ArrayData = Union[NumericalArray, SymbolicArray]
Matrix2dData = Union[Numerical2dMatrix, Symbolic2dMatrix]

# SymbolicVector = Union[
#     Point3,
#     Vector3,
#     Expression,
# ]
#
# ArrayLikeData = Union[Expression, Iterable, np.ndarray]

# RotationData = Union[
#     TransformationMatrix,
#     RotationMatrix,
#     Expression,
#     Quaternion,
#     np.ndarray,
#     ca.SX,
#     Iterable[Iterable[ScalarData]],
# ]

# SymbolicArray = Union[
#     Expression,
#     Point3,
#     Vector3,
#     RotationMatrix
# ]

# TransformationData = Union[
#     TransformationMatrix,
#     RotationMatrix,
#     Expression,
#     np.ndarray,
#     ca.SX,
#     Iterable[Iterable[ScalarData]],
# ]

SpatialType = TypeVar(
    "SpatialType", Point3, Vector3, TransformationMatrix, RotationMatrix, Quaternion
)
AnyCasType = TypeVar(
    "AnyCasType",
    Symbol,
    Expression,
    Point3,
    Vector3,
    TransformationMatrix,
    RotationMatrix,
    Quaternion,
)
