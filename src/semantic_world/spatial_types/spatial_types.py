from __future__ import annotations

import copy
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import (
    Union,
    TYPE_CHECKING,
    Optional,
    Tuple,
    overload,
    Callable,
)

import casadi as ca
import numpy as np

from . import VectorOperationsMixin, limit
from .symbolic_core import SymbolicType, MatrixOperationsMixin
from ..exceptions import WrongDimensionsError

if TYPE_CHECKING:
    from .type_hint_types import *
    from ..world_description.world_entity import KinematicStructureEntity


@dataclass
class ReferenceFrameMixin:
    """
    Provides functionality to associate a reference frame with an object.

    This mixin class allows the inclusion of a reference frame within objects that
    require spatial or kinematic context. The reference frame is represented by a
    `KinematicStructureEntity`, which provides the necessary structural and spatial
    information.

    """

    reference_frame: Optional[KinematicStructureEntity]
    """
    The reference frame associated with the object. Can be None if no reference frame is required or applicable.
    """


def _operation_type_error(arg1: object, operation: str, arg2: object) -> TypeError:
    return TypeError(
        f"unsupported operand type(s) for {operation}: '{arg1.__class__.__name__}' "
        f"and '{arg2.__class__.__name__}'"
    )


@dataclass
class TransformationMatrix(SymbolicType, ReferenceFrameMixin, MatrixOperationsMixin):
    """
    Represents a 4x4 transformation matrix used in kinematics and transformations.

    A `TransformationMatrix` encapsulates relationships between a parent coordinate
    system (reference frame) and a child coordinate system through rotation and
    translation. It provides utilities to derive transformations, compute dot
    products, and create transformations from various inputs such as Euler angles or
    quaternions.
    """

    child_frame: Optional[KinematicStructureEntity]
    """
    The child or target frame associated with the transformation.
    """

    def __post_init__(self):

        self.reference_frame = reference_frame
        self.child_frame = child_frame
        if data is None:
            self.casadi_sx = ca.SX.eye(4)
            return
        elif isinstance(data, ca.SX):
            self.casadi_sx = data
        elif isinstance(data, (Expression, RotationMatrix, TransformationMatrix)):
            self.casadi_sx = copy(data.casadi_sx)
            if isinstance(data, RotationMatrix):
                self.reference_frame = self.reference_frame or data.reference_frame
            if isinstance(data, TransformationMatrix):
                self.child_frame = self.child_frame or data.child_frame
        else:
            self.casadi_sx = Expression(data).casadi_sx
        if sanity_check:
            self._validate()

    def _validate(self):
        if self.shape[0] != 4 or self.shape[1] != 4:
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[3, 0] = 0.0
        self[3, 1] = 0.0
        self[3, 2] = 0.0
        self[3, 3] = 1.0

    @classmethod
    def from_iterable(
        cls,
        data: NumericalData,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        Initializes an instance of the transformation matrix or related object with optional
        data and frame references. Performs optional sanity check validation during initialization.

        :param data: Optional data used to initialize the transformation matrix. It can be
                     of type TransformationData, symbolic expression (ca.SX), Expression,
                     RotationMatrix, or TransformationMatrix.
        :param reference_frame: The reference frame of the transformation.
        :param child_frame: The child frame associated with the transformation.
        :param sanity_check: A boolean indicating whether a validation (sanity check) should
                             be performed during initialization. Defaults to True.
        """
        pass

    @classmethod
    def from_point_rotation_matrix(
        cls,
        point: Optional[Point3] = None,
        rotation_matrix: Optional[RotationMatrix] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> TransformationMatrix:
        """
        Constructs a TransformationMatrix object from a given point, a rotation matrix,
        a reference frame, and a child frame.

        :param point: The 3D point used to set the translation part of the
            transformation matrix. If None, no translation is applied.
        :param rotation_matrix: The rotation matrix defines the rotational component
            of the transformation. If None, the identity matrix is assumed.
        :param reference_frame: The reference frame for the transformation matrix.
            It specifies the parent coordinate system.
        :param child_frame: The child or target frame for the transformation. It
            specifies the target coordinate system.
        :return: A `TransformationMatrix` instance initialized with the provided
            parameters or default values.
        """
        if rotation_matrix is None:
            a_T_b = cls(reference_frame=reference_frame, child_frame=child_frame)
        else:
            a_T_b = cls(
                rotation_matrix,
                reference_frame=reference_frame,
                child_frame=child_frame,
                sanity_check=False,
            )
        if point is not None:
            a_T_b[0, 3] = point.x
            a_T_b[1, 3] = point.y
            a_T_b[2, 3] = point.z
        return a_T_b

    @classmethod
    def from_xyz_rpy(
        cls,
        x: ScalarData = 0,
        y: ScalarData = 0,
        z: ScalarData = 0,
        roll: ScalarData = 0,
        pitch: ScalarData = 0,
        yaw: ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> TransformationMatrix:
        """
        Creates a TransformationMatrix object from position (x, y, z) and Euler angles
        (roll, pitch, yaw) values. The function also accepts optional reference and
        child frame parameters.

        :param x: The x-coordinate of the position
        :param y: The y-coordinate of the position
        :param z: The z-coordinate of the position
        :param roll: The rotation around the x-axis
        :param pitch: The rotation around the y-axis
        :param yaw: The rotation around the z-axis
        :param reference_frame: The reference frame for the transformation
        :param child_frame: The child frame associated with the transformation
        :return: A TransformationMatrix object created using the provided
            position and orientation values
        """
        p = Point3(x, y, z)
        r = RotationMatrix.from_rpy(roll, pitch, yaw)
        return cls.from_point_rotation_matrix(
            p, r, reference_frame=reference_frame, child_frame=child_frame
        )

    @classmethod
    def from_xyz_quaternion(
        cls,
        pos_x: ScalarData = 0,
        pos_y: ScalarData = 0,
        pos_z: ScalarData = 0,
        quat_w: ScalarData = 0,
        quat_x: ScalarData = 0,
        quat_y: ScalarData = 0,
        quat_z: ScalarData = 1,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> TransformationMatrix:
        """
        Creates a `TransformationMatrix` instance from the provided position coordinates and quaternion
        values representing rotation. This method constructs a 3D point for the position and a rotation
        matrix derived from the quaternion, and initializes the transformation matrix with these along
        with optional reference and child frame entities.

        :param pos_x: X coordinate of the position in space.
        :param pos_y: Y coordinate of the position in space.
        :param pos_z: Z coordinate of the position in space.
        :param quat_w: W component of the quaternion representing rotation.
        :param quat_x: X component of the quaternion representing rotation.
        :param quat_y: Y component of the quaternion representing rotation.
        :param quat_z: Z component of the quaternion representing rotation.
        :param reference_frame: Optional reference frame for the transformation matrix.
        :param child_frame: Optional child frame for the transformation matrix.
        :return: A `TransformationMatrix` object constructed from the given parameters.
        """
        p = Point3(pos_x, pos_y, pos_z)
        r = RotationMatrix.from_quaternion(
            q=Quaternion(w=quat_w, x=quat_x, y=quat_y, z=quat_z)
        )
        return cls.from_point_rotation_matrix(
            p, r, reference_frame=reference_frame, child_frame=child_frame
        )

    @property
    def x(self) -> Expression:
        return self[0, 3]

    @x.setter
    def x(self, value: ScalarData):
        self[0, 3] = value

    @property
    def y(self) -> Expression:
        return self[1, 3]

    @y.setter
    def y(self, value: ScalarData):
        self[1, 3] = value

    @property
    def z(self) -> Expression:
        return self[2, 3]

    @z.setter
    def z(self, value: ScalarData):
        self[2, 3] = value

    def dot(self, other: SpatialType) -> SpatialType:
        if isinstance(other, (Vector3, Point3, RotationMatrix, TransformationMatrix)):
            result = ca.mtimes(self.casadi_sx, other.casadi_sx)
            if isinstance(other, Vector3):
                result = Vector3.from_iterable(
                    result, reference_frame=self.reference_frame
                )
                return result
            if isinstance(other, Point3):
                result = Point3.from_iterable(
                    result, reference_frame=self.reference_frame
                )
                return result
            if isinstance(other, RotationMatrix):
                result = RotationMatrix(
                    result, reference_frame=self.reference_frame, sanity_check=False
                )
                return result
            if isinstance(other, TransformationMatrix):
                result = TransformationMatrix(
                    result,
                    reference_frame=self.reference_frame,
                    child_frame=other.child_frame,
                    sanity_check=False,
                )
                return result
        raise _operation_type_error(self, "dot", other)

    def __matmul__(self, other: SpatialType) -> SpatialType:
        return self.dot(other)

    def inverse(self) -> TransformationMatrix:
        inv = TransformationMatrix(
            child_frame=self.reference_frame, reference_frame=self.child_frame
        )
        inv[:3, :3] = self[:3, :3].T
        inv[:3, 3] = (-inv[:3, :3]).dot(self[:3, 3])
        return inv

    def to_position(self) -> Point3:
        result = Point3.from_iterable(
            self[:4, 3:], reference_frame=self.reference_frame
        )
        return result

    def to_translation(self) -> TransformationMatrix:
        """
        :return: sets the rotation part of a frame to identity
        """
        r = TransformationMatrix()
        r[0, 3] = self[0, 3]
        r[1, 3] = self[1, 3]
        r[2, 3] = self[2, 3]
        return TransformationMatrix(
            r, reference_frame=self.reference_frame, child_frame=None
        )

    def to_rotation_matrix(self) -> RotationMatrix:
        return RotationMatrix(self)

    def to_quaternion(self) -> Quaternion:
        return Quaternion.from_rotation_matrix(self)

    def __deepcopy__(self, memo) -> TransformationMatrix:
        """
        Even in a deep copy, we don't want to copy the reference and child frame, just the matrix itself,
        because are just references to kinematic structure entities.
        """
        if id(self) in memo:
            return memo[id(self)]
        return TransformationMatrix(
            deepcopy(self.casadi_sx),
            reference_frame=self.reference_frame,
            child_frame=self.child_frame,
        )


class RotationMatrix(SymbolicType, ReferenceFrameMixin, MatrixOperationsMixin):
    """
    Class to represent a 4x4 symbolic rotation matrix tied to kinematic references.

    This class provides methods for creating and manipulating rotation matrices within the context
    of kinematic structures. It supports initialization using data such as quaternions, axis-angle,
    other matrices, or directly through vector definitions. The primary purpose is to facilitate
    rotational transformations and computations in a symbolic context, particularly for applications
    like robotic kinematics or mechanical engineering.
    """

    child_frame: Optional[KinematicStructureEntity]
    """
    Child kinematic frame associated with this rotation matrix.
    """

    def __init__(
        self,
        data: Optional[RotationData] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
        sanity_check: bool = True,
    ):
        """
        Initializes the instance with rotation, reference frame, and
        child frame, and performs a sanity check to validate the shape.

        :param data: Optional object representing the rotation or transformation. Can be of type
            RotationData (e.g., casadi SX, Quaternion, RotationMatrix, TransformationMatrix) or None.
        :param reference_frame: Optional KinematicStructureEntity instance representing the
            reference frame of the rotation or transformation.
        :param child_frame: Optional KinematicStructureEntity instance representing the child
            frame of the rotation or transformation.
        :param sanity_check: Boolean flag indicating whether to perform a sanity check to validate
            the shape of the input data. Defaults to True.

        :raises ValueError: If sanity_check is True and the shape of the input data is not 4x4.
        """
        self.reference_frame = reference_frame
        self.child_frame = child_frame
        if isinstance(data, ca.SX):
            self.casadi_sx = data
        elif isinstance(data, Quaternion):
            self.casadi_sx = self.__quaternion_to_rotation_matrix(data).casadi_sx
            self.reference_frame = self.reference_frame or data.reference_frame
        elif isinstance(data, (RotationMatrix, TransformationMatrix)):
            self.casadi_sx = copy(data.casadi_sx)
            self.reference_frame = data.reference_frame
            self.child_frame = child_frame
        elif data is None:
            self.casadi_sx = ca.SX.eye(4)
            return
        else:
            self.casadi_sx = Expression(data).casadi_sx
        if sanity_check:
            if self.shape[0] != 4 or self.shape[1] != 4:
                raise WrongDimensionsError(
                    expected_dimensions=(4, 4), actual_dimensions=self.shape
                )
            self[0, 3] = 0
            self[1, 3] = 0
            self[2, 3] = 0
            self[3, 0] = 0
            self[3, 1] = 0
            self[3, 2] = 0
            self[3, 3] = 1

    @classmethod
    def from_axis_angle(
        cls,
        axis: Vector3,
        angle: ScalarData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> RotationMatrix:
        """
        Conversion of unit axis and angle to 4x4 rotation matrix according to:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        """
        # use casadi to prevent a bunch of Expression.__init__.py calls
        axis = _to_sx(axis)
        angle = _to_sx(angle)
        ct = ca.cos(angle)
        st = ca.sin(angle)
        vt = 1 - ct
        m_vt = axis * vt
        m_st = axis * st
        m_vt_0_ax = (m_vt[0] * axis)[1:]
        m_vt_1_2 = m_vt[1] * axis[2]
        s = ca.SX.eye(4)
        ct__m_vt__axis = ct + m_vt * axis
        s[0, 0] = ct__m_vt__axis[0]
        s[0, 1] = -m_st[2] + m_vt_0_ax[0]
        s[0, 2] = m_st[1] + m_vt_0_ax[1]
        s[1, 0] = m_st[2] + m_vt_0_ax[0]
        s[1, 1] = ct__m_vt__axis[1]
        s[1, 2] = -m_st[0] + m_vt_1_2
        s[2, 0] = -m_st[1] + m_vt_0_ax[1]
        s[2, 1] = m_st[0] + m_vt_1_2
        s[2, 2] = ct__m_vt__axis[2]
        return cls(s, reference_frame=reference_frame, sanity_check=False)

    @classmethod
    def __quaternion_to_rotation_matrix(cls, q: Quaternion) -> RotationMatrix:
        """
        Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        return cls(
            [
                [w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                [0, 0, 0, 1],
            ],
            reference_frame=q.reference_frame,
        )

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> RotationMatrix:
        return cls.__quaternion_to_rotation_matrix(q)

    def x_vector(self) -> Vector3:
        return Vector3(
            x=self[0, 0],
            y=self[1, 0],
            z=self[2, 0],
            reference_frame=self.reference_frame,
        )

    def y_vector(self) -> Vector3:
        return Vector3(
            x=self[0, 1],
            y=self[1, 1],
            z=self[2, 1],
            reference_frame=self.reference_frame,
        )

    def z_vector(self) -> Vector3:
        return Vector3(
            x=self[0, 2],
            y=self[1, 2],
            z=self[2, 2],
            reference_frame=self.reference_frame,
        )

    @overload
    def dot(self, other: Vector3) -> Vector3: ...

    @overload
    def dot(self, other: RotationMatrix) -> RotationMatrix: ...

    @overload
    def dot(self, other: TransformationMatrix) -> TransformationMatrix: ...

    def dot(self, other):
        if isinstance(other, (Vector3, RotationMatrix, TransformationMatrix)):
            result = ca.mtimes(self.casadi_sx, other.casadi_sx)
            if isinstance(other, Vector3):
                result = Vector3.from_iterable(result)
            elif isinstance(other, RotationMatrix):
                result = RotationMatrix(result, sanity_check=False)
            elif isinstance(other, TransformationMatrix):
                result = TransformationMatrix(result, sanity_check=False)
            result.reference_frame = self.reference_frame
            return result
        raise _operation_type_error(self, "dot", other)

    @overload
    def __matmul__(self, other: Vector3) -> Vector3: ...

    @overload
    def __matmul__(self, other: RotationMatrix) -> RotationMatrix: ...

    @overload
    def __matmul__(self, other: TransformationMatrix) -> TransformationMatrix: ...

    def __matmul__(self, other):
        return self.dot(other)

    def to_axis_angle(self) -> Tuple[Vector3, Expression]:
        return self.to_quaternion().to_axis_angle()

    def to_angle(self, hint: Optional[Callable] = None) -> Expression:
        """
        :param hint: A function whose sign of the result will be used to determine if angle should be positive or
                        negative
        :return:
        """
        axis, angle = self.to_axis_angle()
        if hint is not None:
            return normalize_angle(
                if_greater_zero(hint(axis), if_result=angle, else_result=-angle)
            )
        else:
            return angle

    @classmethod
    def from_vectors(
        cls,
        x: Optional[Vector3] = None,
        y: Optional[Vector3] = None,
        z: Optional[Vector3] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> RotationMatrix:
        """
        Create a rotation matrix from 2 or 3 orthogonal vectors.

        If exactly two of x, y, z must be provided. The third will be computed using the cross product.

        Valid combinations:
        - x and y provided: z = x × y
        - x and z provided: y = z × x
        - y and z provided: x = y × z
        - x, y, and z provided: all three used directly
        """

        if x is not None and y is not None and z is None:
            z = x.cross(y)
        elif x is not None and y is None and z is not None:
            y = z.cross(x)
        elif x is None and y is not None and z is not None:
            x = y.cross(z)
        x.scale(1)
        y.scale(1)
        z.scale(1)
        R = cls(
            [
                [x[0], y[0], z[0], 0],
                [x[1], y[1], z[1], 0],
                [x[2], y[2], z[2], 0],
                [0, 0, 0, 1],
            ],
            reference_frame=reference_frame,
        )
        return R

    @classmethod
    def from_rpy(
        cls,
        roll: Optional[ScalarData] = None,
        pitch: Optional[ScalarData] = None,
        yaw: Optional[ScalarData] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> RotationMatrix:
        """
        Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        roll = 0 if roll is None else roll
        pitch = 0 if pitch is None else pitch
        yaw = 0 if yaw is None else yaw
        try:
            roll = roll.casadi_sx
        except AttributeError:
            pass
        try:
            pitch = pitch.casadi_sx
        except AttributeError:
            pass
        try:
            yaw = yaw.casadi_sx
        except AttributeError:
            pass
        s = ca.SX.eye(4)

        s[0, 0] = ca.cos(yaw) * ca.cos(pitch)
        s[0, 1] = (ca.cos(yaw) * ca.sin(pitch) * ca.sin(roll)) - (
            ca.sin(yaw) * ca.cos(roll)
        )
        s[0, 2] = (ca.sin(yaw) * ca.sin(roll)) + (
            ca.cos(yaw) * ca.sin(pitch) * ca.cos(roll)
        )
        s[1, 0] = ca.sin(yaw) * ca.cos(pitch)
        s[1, 1] = (ca.cos(yaw) * ca.cos(roll)) + (
            ca.sin(yaw) * ca.sin(pitch) * ca.sin(roll)
        )
        s[1, 2] = (ca.sin(yaw) * ca.sin(pitch) * ca.cos(roll)) - (
            ca.cos(yaw) * ca.sin(roll)
        )
        s[2, 0] = -ca.sin(pitch)
        s[2, 1] = ca.cos(pitch) * ca.sin(roll)
        s[2, 2] = ca.cos(pitch) * ca.cos(roll)
        return cls(s, reference_frame=reference_frame, sanity_check=False)

    def inverse(self) -> RotationMatrix:
        return self.T

    def to_rpy(self) -> Tuple[Expression, Expression, Expression]:
        """
        :return: roll, pitch, yaw
        """
        i = 0
        j = 1
        k = 2

        cy = sqrt(self[i, i] * self[i, i] + self[j, i] * self[j, i])
        if0 = cy - _EPS
        ax = if_greater_zero(
            if0, atan2(self[k, j], self[k, k]), atan2(-self[j, k], self[j, j])
        )
        ay = if_greater_zero(if0, atan2(-self[k, i], cy), atan2(-self[k, i], cy))
        az = if_greater_zero(if0, atan2(self[j, i], self[i, i]), Expression(0))
        return ax, ay, az

    def to_quaternion(self) -> Quaternion:
        return Quaternion.from_rotation_matrix(self)

    def normalize(self) -> None:
        """Scales each of the axes to the length of one."""
        scale_v = 1.0
        self[:3, 0] = self[:3, 0].scale(scale_v)
        self[:3, 1] = self[:3, 1].scale(scale_v)
        self[:3, 2] = self[:3, 2].scale(scale_v)

    @property
    def T(self) -> RotationMatrix:
        return RotationMatrix(self.casadi_sx.T, reference_frame=self.reference_frame)

    def rotational_error(self, other: RotationMatrix) -> Expression:
        """
        Calculate the rotational error between two rotation matrices.

        This function computes the angular difference between two rotation matrices
        by computing the dot product of the first matrix and the inverse of the second.
        Subsequently, it generates the angle of the resulting rotation matrix.

        :param other: The second rotation matrix.
        :return: The angular error between the two rotation matrices as an expression.
        """
        r_distance = self.dot(other.inverse())
        return r_distance.to_angle()


class Point3(SymbolicType, ReferenceFrameMixin):
    """
    Represents a 3D point with reference frame handling.

    This class provides a representation of a point in 3D space, including support
    for operations such as addition, subtraction, projection onto planes/lines, and
    distance calculations. It incorporates a reference frame for kinematic computations
    and facilitates mathematical operations essential for 3D geometry modeling.

    Note that it is represented as a 4d vector, where the last entry is always a 1.
    """

    def __init__(
        self,
        x: ScalarData = 0,
        y: ScalarData = 0,
        z: ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        Represents an initialization for a kinematic structure entity with specified
        positional components and a reference frame.

        :param x: Position component along the x-axis.
        :param y: Position component along the y-axis.
        :param z: Position component along the z-axis.
        :param reference_frame: Reference frame entity associated with this instance.
        """
        self.reference_frame = reference_frame
        # casadi can't be initialized with an array that mixes int/float and SX
        self.casadi_sx = ca.SX([0, 0, 0, 1])
        self[0] = x
        self[1] = y
        self[2] = z

    @classmethod
    def from_iterable(
        cls,
        data: Union[ArrayLikeData, Vector3, Point3],
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Point3:
        """
        Creates an instance of Point3 from provided iterable data.

        This class method is used to construct a Point3 object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Point3
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Point3 instance.
        :param reference_frame: A reference to a `KinematicStructureEntity` object,
            representing the frame of reference for the Point3 instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified,
            it will be taken from the data.
        :return: Returns an instance of Point3 initialized with the processed data
            and an optional reference frame.
        """
        if isinstance(data, (Quaternion, RotationMatrix, TransformationMatrix)):
            raise TypeError(f"Can't create a Point3 form {type(data)}")
        if hasattr(data, "shape") and len(data.shape) > 1 and data.shape[1] != 1:
            raise ValueError("The iterable must be a 1d list, tuple or array")
        if hasattr(data, "reference_frame") and reference_frame is None:
            reference_frame = data.reference_frame
        return cls(data[0], data[1], data[2], reference_frame=reference_frame)

    def norm(self) -> Expression:
        return Expression(ca.norm_2(self[:3].casadi_sx))

    @property
    def x(self) -> Expression:
        return self[0]

    @x.setter
    def x(self, value: ScalarData):
        self[0] = value

    @property
    def y(self) -> Expression:
        return self[1]

    @y.setter
    def y(self, value: ScalarData):
        self[1] = value

    @property
    def z(self) -> Expression:
        return self[2]

    @z.setter
    def z(self, value: ScalarData):
        self[2] = value

    def __add__(self, other: Vector3) -> Point3:
        if isinstance(other, Vector3):
            result = Point3.from_iterable(self.casadi_sx.__add__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    @overload
    def __sub__(self, other: Point3) -> Vector3: ...

    @overload
    def __sub__(self, other: Vector3) -> Point3: ...

    def __sub__(self, other):
        if isinstance(other, Point3):
            result = Vector3.from_iterable(self.casadi_sx.__sub__(other.casadi_sx))
        elif isinstance(other, Vector3):
            result = Point3.from_iterable(self.casadi_sx.__sub__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __neg__(self) -> Point3:
        result = Point3.from_iterable(self.casadi_sx.__neg__())
        result.reference_frame = self.reference_frame
        return result

    def project_to_plane(
        self, frame_V_plane_vector1: Vector3, frame_V_plane_vector2: Vector3
    ) -> Tuple[Point3, Expression]:
        """
        Projects a point onto a plane defined by two vectors.
        This function assumes that all parameters are defined with respect to the same reference frame.

        :param frame_V_plane_vector1: First vector defining the plane
        :param frame_V_plane_vector2: Second vector defining the plane
        :return: Tuple of (projected point on the plane, signed distance from point to plane)
        """
        normal = frame_V_plane_vector1.cross(frame_V_plane_vector2)
        normal.scale(1)
        frame_V_current = Vector3.from_iterable(self)
        d = normal @ frame_V_current
        projection = self - normal * d
        return projection, d

    def project_to_line(
        self, line_point: Point3, line_direction: Vector3
    ) -> Tuple[Point3, Expression]:
        """
        :param line_point: a point that the line intersects, must have the same reference frame as self
        :param line_direction: the direction of the line, must have the same reference frame as self
        :return: tuple of (closest point on the line, shortest distance between self and the line)
        """
        lp_vector = self - line_point
        cross_product = lp_vector.cross(line_direction)
        distance = cross_product.norm() / line_direction.norm()

        line_direction_unit = line_direction / line_direction.norm()
        projection_length = lp_vector @ line_direction_unit
        closest_point = line_point + line_direction_unit * projection_length

        return closest_point, distance

    def distance_to_line_segment(
        self, line_start: Point3, line_end: Point3
    ) -> Tuple[Expression, Point3]:
        """
        All parameters must have the same reference frame as self.
        :param line_start: start of the approached line
        :param line_end: end of the approached line
        :return: distance to line, the nearest point on the line
        """
        frame_P_current = self
        frame_P_line_start = line_start
        frame_P_line_end = line_end
        frame_V_line_vec = frame_P_line_end - frame_P_line_start
        pnt_vec = frame_P_current - frame_P_line_start
        line_len = frame_V_line_vec.norm()
        line_unitvec = frame_V_line_vec / line_len
        pnt_vec_scaled = pnt_vec / line_len
        t = line_unitvec @ pnt_vec_scaled
        t = limit(t, lower_limit=0.0, upper_limit=1.0)
        frame_V_offset = frame_V_line_vec * t
        dist = (frame_V_offset - pnt_vec).norm()
        frame_P_nearest = frame_P_line_start + frame_V_offset
        return dist, frame_P_nearest


class Vector3(SymbolicType, ReferenceFrameMixin, VectorOperationsMixin):
    """
    Representation of a 3D vector with reference frame support for homogenous transformations.

    This class provides a structured representation of 3D vectors. It includes
    support for operations such as addition, subtraction, scaling, dot product,
    cross product, and more. It is compatible with symbolic computations and
    provides methods to define standard basis vectors, normalize a vector, and
    compute geometric properties such as the angle between vectors. The class
    also includes support for working in different reference frames.

    Note that it is represented as a 4d vector, where the last entry is always a 0.
    """

    vis_frame: Optional[KinematicStructureEntity]
    """
    The reference frame associated with the vector, used for visualization purposes only. Optional.
    It will be visualized at the origin of the vis_frame
    """

    def __init__(
        self,
        x: ScalarData = 0,
        y: ScalarData = 0,
        z: ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        Initializes a 3D point with optional reference frame.

        This constructor creates a 3D point represented by `x`, `y`, and `z` coordinates
        within a given reference frame. If no reference frame is specified, a default
        frame will be assumed. The internal representation of the point is stored as a
        `Point3` object, and its scalar coordinates and reference frame are then
        directly accessible.

        :param x: X-coordinate of the point. Defaults to 0.
        :param y: Y-coordinate of the point. Defaults to 0.
        :param z: Z-coordinate of the point. Defaults to 0.
        :param reference_frame: Optional reference frame for the point. If not provided, defaults to None.
        """
        point = Point3(x, y, z, reference_frame=reference_frame)
        self.casadi_sx = point.s
        self.reference_frame = point.reference_frame
        self.vis_frame = self.reference_frame
        self[3] = 0

    @classmethod
    def from_iterable(
        cls,
        data: Union[ArrayLikeData, Vector3, Point3],
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Vector3:
        """
        Creates an instance of Vector3 from provided iterable data.

        This class method is used to construct a Vector3 object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Vector3
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Vector3 instance.
        :param reference_frame: A reference to a `KinematicStructureEntity` object,
            representing the frame of reference for the Vector3 instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified,
            it will be taken from the data.
        :return: Returns an instance of Vector3 initialized with the processed data
            and an optional reference frame.
        """
        if isinstance(data, (Quaternion, RotationMatrix, TransformationMatrix)):
            raise TypeError(f"Can't create a Vector3 form {type(data)}")
        if hasattr(data, "shape") and len(data.shape) > 1 and data.shape[1] != 1:
            raise ValueError("The iterable must be a 1d list, tuple or array")
        if hasattr(data, "reference_frame") and reference_frame is None:
            reference_frame = data.reference_frame
        result = cls(data[0], data[1], data[2], reference_frame=reference_frame)
        if hasattr(data, "vis_frame"):
            result.vis_frame = data.vis_frame
        return result

    @classmethod
    def X(cls, reference_frame: Optional[KinematicStructureEntity] = None) -> Vector3:
        return cls(x=1, y=0, z=0, reference_frame=reference_frame)

    @classmethod
    def Y(cls, reference_frame: Optional[KinematicStructureEntity] = None) -> Vector3:
        return cls(x=0, y=1, z=0, reference_frame=reference_frame)

    @classmethod
    def Z(cls, reference_frame: Optional[KinematicStructureEntity] = None) -> Vector3:
        return cls(x=0, y=0, z=1, reference_frame=reference_frame)

    @classmethod
    def unit_vector(
        cls,
        x: ScalarData = 0,
        y: ScalarData = 0,
        z: ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Vector3:
        v = cls(x, y, z, reference_frame=reference_frame)
        v.scale(1, unsafe=True)
        return v

    @property
    def x(self) -> Expression:
        return self[0]

    @x.setter
    def x(self, value: ScalarData):
        self[0] = value

    @property
    def y(self) -> Expression:
        return self[1]

    @y.setter
    def y(self, value: ScalarData):
        self[1] = value

    @property
    def z(self) -> Expression:
        return self[2]

    @z.setter
    def z(self, value: ScalarData):
        self[2] = value

    def __add__(self, other: Vector3) -> Vector3:
        if isinstance(other, Vector3):
            result = Vector3.from_iterable(self.casadi_sx.__add__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __sub__(self, other: Vector3) -> Vector3:
        if isinstance(other, Vector3):
            result = Vector3.from_iterable(self.casadi_sx.__sub__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __mul__(self, other: ScalarData) -> Vector3:
        if isinstance(other, (int, float, Symbol, Expression)):
            result = Vector3.from_iterable(self.casadi_sx.__mul__(_to_sx(other)))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __rmul__(self, other: float) -> Vector3:
        if isinstance(other, (int, float)):
            result = Vector3.from_iterable(self.casadi_sx.__mul__(other))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __truediv__(self, other: ScalarData) -> Vector3:
        if isinstance(other, (int, float, Symbol, Expression)):
            result = Vector3.from_iterable(self.casadi_sx.__truediv__(_to_sx(other)))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def safe_division(
        self,
        other: ScalarData,
        if_nan: Optional[Vector3] = None,
    ) -> AnyCasType:
        """
        A version of division where no sub-expression is ever NaN. The expression would evaluate to 'if_nan', but
        you should probably never work with the 'if_nan' result. However, if one sub-expressions is NaN, the whole expression
        evaluates to NaN, even if it is only in a branch of an if-else, that is not returned.
        This method is a workaround for such cases.
        """
        if if_nan is None:
            if_nan = Vector3()
        save_denominator = if_eq_zero(
            condition=other, if_result=Expression(1), else_result=other
        )
        return if_eq_zero(other, if_result=if_nan, else_result=self / save_denominator)

    def __neg__(self) -> Vector3:
        result = Vector3.from_iterable(self.casadi_sx.__neg__())
        result.reference_frame = self.reference_frame
        return result

    def dot(self, other: Vector3) -> Expression:
        if isinstance(other, Vector3):
            return Expression(ca.mtimes(self[:3].T.casadi_sx, other[:3].casadi_sx))
        raise _operation_type_error(self, "dot", other)

    def __matmul__(self, other: Vector3) -> Expression:
        return self.dot(other)

    def cross(self, other: Vector3) -> Vector3:
        result = ca.cross(self.casadi_sx[:3], other.casadi_sx[:3])
        result = self.__class__.from_iterable(result)
        result.reference_frame = self.reference_frame
        return result

    def norm(self) -> Expression:
        return Expression(ca.norm_2(self[:3].casadi_sx))

    def scale(self, a: ScalarData, unsafe: bool = False):
        if unsafe:
            self.casadi_sx = ((self / self.norm()) * a).casadi_sx
        else:
            self.casadi_sx = (self.safe_division(self.norm()) * a).casadi_sx

    def project_to_cone(
        self, frame_V_cone_axis: Vector3, cone_theta: Union[Symbol, float, Expression]
    ) -> Vector3:
        """
        Projects a given vector onto the boundary of a cone defined by its axis and angle.

        This function computes the projection of a vector onto the boundary of a
        cone specified by its axis and half-angle. It handles special cases where
        the input vector is collinear with the cone's axis. The projection ensures
        the resulting vector lies within the cone's boundary.

        :param frame_V_current: The vector to be projected.
        :param frame_V_cone_axis: The axis of the cone.
        :param cone_theta: The half-angle of the cone in radians. Can be a symbolic value or a float.
        :return: The projection of the input vector onto the cone's boundary.
        """
        frame_V_current = self
        frame_V_cone_axis_normed = Vector3.from_iterable(frame_V_cone_axis)
        frame_V_cone_axis_normed.scale(1)
        beta = frame_V_current @ frame_V_cone_axis_normed
        norm_v = frame_V_current.norm()

        # Compute the perpendicular component.
        v_perp = frame_V_current - (frame_V_cone_axis_normed * beta)
        norm_v_perp = v_perp.norm()
        v_perp.scale(1)

        s = beta * cos(cone_theta) + norm_v_perp * sin(cone_theta)
        projected_vector = (
            (frame_V_cone_axis_normed * cos(cone_theta)) + (v_perp * sin(cone_theta))
        ) * s
        # Handle the case when v is collinear with a.
        project_on_cone_boundary = if_less(
            a=norm_v_perp,
            b=1e-8,
            if_result=frame_V_cone_axis_normed * norm_v * cos(cone_theta),
            else_result=projected_vector,
        )

        return if_greater_eq(
            a=beta,
            b=norm_v * np.cos(cone_theta),
            if_result=frame_V_current,
            else_result=project_on_cone_boundary,
        )

    def angle_between(self, other: Vector3) -> Expression:
        return acos(
            limit(
                self @ other / (self.norm() * other.norm()),
                lower_limit=-1,
                upper_limit=1,
            )
        )

    def slerp(self, other: Vector3, t: ScalarData) -> Vector3:
        """
        spherical linear interpolation
        :param other: vector of same length as self
        :param t: value between 0 and 1. 0 is v1 and 1 is v2
        """
        angle = safe_acos(self @ other)
        angle2 = if_eq(angle, 0, Expression(1), angle)
        return if_eq(
            angle,
            0,
            self,
            self * (sin((1 - t) * angle2) / sin(angle2))
            + other * (sin(t * angle2) / sin(angle2)),
        )


class Quaternion(SymbolicType, ReferenceFrameMixin):
    """
    Represents a quaternion, which is a mathematical entity used to encode
    rotations in three-dimensional space.

    The Quaternion class provides methods for creating quaternion objects
    from various representations, such as axis-angle, roll-pitch-yaw,
    and rotation matrices. It supports operations to define and manipulate
    rotations in 3D space efficiently. Quaternions are used extensively
    in physics, computer graphics, robotics, and aerospace engineering
    to represent orientations and rotations.
    """

    def __init__(
        self,
        x: ScalarData = 0.0,
        y: ScalarData = 0.0,
        z: ScalarData = 0.0,
        w: ScalarData = 1.0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        Initializes a new instance of the class with scalar components x, y, z, w along
        with an optional reference frame.

        This constructor ensures that the provided x, y, z, w values are scalar data. If
        they are not scalar values or their shapes are incompatible, a ValueError will
        be raised.

        :param x: The x component of the scalar data. Default is 0.0.
        :param y: The y component of the scalar data. Default is 0.0.
        :param z: The z component of the scalar data. Default is 0.0.
        :param w: The w component of the scalar data. Default is 1.0.
        :param reference_frame: The reference frame entity for the instance. Default is None.

        :raises ValueError: If the shapes of x, y, z, or w are not scalars or their
              shapes are incompatible as per allowed dimensions.
        """
        if hasattr(x, "shape") and x.shape not in (tuple(), (1, 1)):
            raise ValueError("x, y, z, w must be scalars")
        self.reference_frame = reference_frame
        self.casadi_sx = ca.SX(4, 1)
        self[0], self[1], self[2], self[3] = x, y, z, w

    def __neg__(self) -> Quaternion:
        return Quaternion.from_iterable(self.casadi_sx.__neg__())

    @classmethod
    def from_iterable(
        cls,
        data: Union[ArrayLikeData, Quaternion],
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Quaternion:
        """
        Creates an instance of Quaternion from provided iterable data.

        This class method is used to construct a Quaternion object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Quaternion
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Quaternion instance.
        :param reference_frame: A reference to a `KinematicStructureEntity` object,
            representing the frame of reference for the Quaternion instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified,
            it will be taken from the data.

        :return: Returns an instance of Quaternion initialized with the processed data
            and an optional reference frame.
        """
        if isinstance(data, (Point3, Vector3, RotationMatrix, TransformationMatrix)):
            raise TypeError(f"Can't create a Quaternion form {type(data)}")
        if hasattr(data, "shape") and len(data.shape) > 1 and data.shape[1] != 1:
            raise ValueError("The iterable must be a 1d list, tuple or array")
        if hasattr(data, "reference_frame") and reference_frame is None:
            reference_frame = data.reference_frame
        return cls(data[0], data[1], data[2], data[3], reference_frame=reference_frame)

    @property
    def x(self) -> Expression:
        return self[0]

    @x.setter
    def x(self, value: ScalarData):
        self[0] = value

    @property
    def y(self) -> Expression:
        return self[1]

    @y.setter
    def y(self, value: ScalarData):
        self[1] = value

    @property
    def z(self) -> Expression:
        return self[2]

    @z.setter
    def z(self, value: ScalarData):
        self[2] = value

    @property
    def w(self) -> Expression:
        return self[3]

    @w.setter
    def w(self, value: ScalarData):
        self[3] = value

    @classmethod
    def from_axis_angle(
        cls,
        axis: Vector3,
        angle: ScalarData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Quaternion:
        """
        Creates a quaternion from an axis-angle representation.

        This method uses the axis of rotation and the rotation angle (in radians)
        to construct a quaternion representation of the rotation. Optionally,
        a reference frame can be specified to which the resulting quaternion is
        associated.

        :param axis: A 3D vector representing the axis of rotation.
        :param angle: The rotation angle in radians.
        :param reference_frame: An optional reference frame entity associated
            with the quaternion, if applicable.
        :return: A quaternion representing the rotation defined by
            the given axis and angle.
        """
        half_angle = angle / 2
        return cls(
            axis[0] * sin(half_angle),
            axis[1] * sin(half_angle),
            axis[2] * sin(half_angle),
            cos(half_angle),
            reference_frame=reference_frame,
        )

    @classmethod
    def from_rpy(
        cls,
        roll: ScalarData,
        pitch: ScalarData,
        yaw: ScalarData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Quaternion:
        """
        Creates a Quaternion instance from specified roll, pitch, and yaw angles.

        The method computes the quaternion representation of the given roll, pitch,
        and yaw angles using trigonometric transformations based on their
        half-angle values for efficient calculations.

        :param roll: The roll angle in radians.
        :param pitch: The pitch angle in radians.
        :param yaw: The yaw angle in radians.
        :param reference_frame: Optional reference frame entity associated with
            the quaternion.
        :return: A Quaternion instance representing the rotation defined by the
            specified roll, pitch, and yaw angles.
        """
        roll = _to_sx(roll)
        pitch = _to_sx(pitch)
        yaw = _to_sx(yaw)
        roll_half = roll / 2.0
        pitch_half = pitch / 2.0
        yaw_half = yaw / 2.0

        c_roll = cos(roll_half)
        s_roll = sin(roll_half)
        c_pitch = cos(pitch_half)
        s_pitch = sin(pitch_half)
        c_yaw = cos(yaw_half)
        s_yaw = sin(yaw_half)

        cc = c_roll * c_yaw
        cs = c_roll * s_yaw
        sc = s_roll * c_yaw
        ss = s_roll * s_yaw

        x = c_pitch * sc - s_pitch * cs
        y = c_pitch * ss + s_pitch * cc
        z = c_pitch * cs - s_pitch * sc
        w = c_pitch * cc + s_pitch * ss

        return cls(x, y, z, w, reference_frame=reference_frame)

    @classmethod
    def from_rotation_matrix(
        cls, r: Union[RotationMatrix, TransformationMatrix]
    ) -> Quaternion:
        """
        Creates a Quaternion object initialized from a given rotation matrix.

        This method constructs a quaternion representation of the provided rotation matrix. It is designed to handle
        different cases of rotation matrix configurations to ensure numerical stability during computation. The resultant
        quaternion adheres to the expected mathematical relationship with the given rotation matrix.

        :param r: The input matrix representing a rotation. It can be either a `RotationMatrix` or `TransformationMatrix`.
                  This matrix is expected to have a valid mathematical structure typical for rotation matrices.

        :return: A new instance of `Quaternion` corresponding to the given rotation matrix `r`.
        """
        q = Expression((0, 0, 0, 0))
        t = r.trace()

        if0 = t - r[3, 3]

        if1 = r[1, 1] - r[0, 0]

        m_i_i = if_greater_zero(if1, r[1, 1], r[0, 0])
        m_i_j = if_greater_zero(if1, r[1, 2], r[0, 1])
        m_i_k = if_greater_zero(if1, r[1, 0], r[0, 2])

        m_j_i = if_greater_zero(if1, r[2, 1], r[1, 0])
        m_j_j = if_greater_zero(if1, r[2, 2], r[1, 1])
        m_j_k = if_greater_zero(if1, r[2, 0], r[1, 2])

        m_k_i = if_greater_zero(if1, r[0, 1], r[2, 0])
        m_k_j = if_greater_zero(if1, r[0, 2], r[2, 1])
        m_k_k = if_greater_zero(if1, r[0, 0], r[2, 2])

        if2 = r[2, 2] - m_i_i

        m_i_i = if_greater_zero(if2, r[2, 2], m_i_i)
        m_i_j = if_greater_zero(if2, r[2, 0], m_i_j)
        m_i_k = if_greater_zero(if2, r[2, 1], m_i_k)

        m_j_i = if_greater_zero(if2, r[0, 2], m_j_i)
        m_j_j = if_greater_zero(if2, r[0, 0], m_j_j)
        m_j_k = if_greater_zero(if2, r[0, 1], m_j_k)

        m_k_i = if_greater_zero(if2, r[1, 2], m_k_i)
        m_k_j = if_greater_zero(if2, r[1, 0], m_k_j)
        m_k_k = if_greater_zero(if2, r[1, 1], m_k_k)

        t = if_greater_zero(if0, t, m_i_i - (m_j_j + m_k_k) + r[3, 3])
        q[0] = if_greater_zero(
            if0,
            r[2, 1] - r[1, 2],
            if_greater_zero(if2, m_i_j + m_j_i, if_greater_zero(if1, m_k_i + m_i_k, t)),
        )
        q[1] = if_greater_zero(
            if0,
            r[0, 2] - r[2, 0],
            if_greater_zero(if2, m_k_i + m_i_k, if_greater_zero(if1, t, m_i_j + m_j_i)),
        )
        q[2] = if_greater_zero(
            if0,
            r[1, 0] - r[0, 1],
            if_greater_zero(if2, t, if_greater_zero(if1, m_i_j + m_j_i, m_k_i + m_i_k)),
        )
        q[3] = if_greater_zero(if0, t, m_k_j - m_j_k)

        q *= 0.5 / sqrt(t * r[3, 3])
        return cls.from_iterable(q, reference_frame=r.reference_frame)

    def conjugate(self) -> Quaternion:
        return Quaternion(
            x=-self[0],
            y=-self[1],
            z=-self[2],
            w=self[3],
            reference_frame=self.reference_frame,
        )

    def multiply(self, q: Quaternion) -> Quaternion:
        return Quaternion(
            x=self.x * q.w + self.y * q.z - self.z * q.y + self.w * q.x,
            y=-self.x * q.z + self.y * q.w + self.z * q.x + self.w * q.y,
            z=self.x * q.y - self.y * q.x + self.z * q.w + self.w * q.z,
            w=-self.x * q.x - self.y * q.y - self.z * q.z + self.w * q.w,
            reference_frame=self.reference_frame,
        )

    def diff(self, q: Quaternion) -> Quaternion:
        """
        :return: quaternion p, such that self*p=q
        """
        return self.conjugate().multiply(q)

    def normalize(self) -> None:
        norm_ = self.norm()
        self.x /= norm_
        self.y /= norm_
        self.z /= norm_
        self.w /= norm_

    def to_axis_angle(self) -> Tuple[Vector3, Expression]:
        self.normalize()
        w2 = sqrt(1 - self.w**2)
        m = if_eq_zero(w2, Expression(1), w2)  # avoid /0
        angle = if_eq_zero(w2, Expression(0), (2 * acos(limit(self.w, -1, 1))))
        x = if_eq_zero(w2, Expression(0), self.x / m)
        y = if_eq_zero(w2, Expression(0), self.y / m)
        z = if_eq_zero(w2, Expression(1), self.z / m)
        return Vector3(x, y, z, reference_frame=self.reference_frame), angle

    def to_rotation_matrix(self) -> RotationMatrix:
        return RotationMatrix.from_quaternion(self)

    def to_rpy(self) -> Tuple[Expression, Expression, Expression]:
        return self.to_rotation_matrix().to_rpy()

    def dot(self, other: Quaternion) -> Expression:
        if isinstance(other, Quaternion):
            return Expression(ca.mtimes(self.casadi_sx.T, other.casadi_sx))
        return NotImplemented

    def slerp(self, other: Quaternion, t: ScalarData) -> Quaternion:
        """
        spherical linear interpolation that takes into account that q == -q
        :param q1: 4x1 Matrix
        :param q2: 4x1 Matrix
        :param t: float, 0-1
        :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
        """
        cos_half_theta = self.dot(other)

        if0 = -cos_half_theta
        other = if_greater_zero(if0, -other, other)
        cos_half_theta = if_greater_zero(if0, -cos_half_theta, cos_half_theta)

        if1 = abs(cos_half_theta) - 1.0

        # enforce acos(x) with -1 < x < 1
        cos_half_theta = min(1, cos_half_theta)
        cos_half_theta = max(-1, cos_half_theta)

        half_theta = acos(cos_half_theta)

        sin_half_theta = sqrt(1.0 - cos_half_theta * cos_half_theta)
        if2 = 0.001 - abs(sin_half_theta)

        ratio_a = (sin((1.0 - t) * half_theta)).safe_division(sin_half_theta)
        ratio_b = sin(t * half_theta).safe_division(sin_half_theta)

        mid_quaternion = Quaternion.from_iterable(
            Expression(self) * 0.5 + Expression(other) * 0.5
        )
        slerped_quaternion = Quaternion.from_iterable(
            Expression(self) * ratio_a + Expression(other) * ratio_b
        )

        return Quaternion.from_iterable(
            if_greater_eq_zero(
                if1, self, if_greater_zero(if2, mid_quaternion, slerped_quaternion)
            )
        )
