from __future__ import annotations

import functools
from enum import IntEnum
from typing import overload, Union, Iterable, Tuple, Optional, Callable, List, Sequence, Dict, TypeVar, \
    TYPE_CHECKING
import numpy as np
import casadi as ca

from scipy import sparse as sp

if TYPE_CHECKING:
    from ..world_entity import Body

all_expressions = Union[Symbol_, Symbol, Expression, Point3, Vector3, RotationMatrix, TransformationMatrix, Quaternion]
all_expressions_float = Union[Symbol, Expression, Point3, Vector3, RotationMatrix, TransformationMatrix, float, Quaternion]
symbol_expr_float = Union[Symbol, Expression, float, int, IntEnum]
symbol_expr = Union[Symbol, Expression]
PreservedCasType = TypeVar('PreservedCasType', Point3, Vector3, TransformationMatrix, RotationMatrix, Quaternion, Expression)

pi: float






















































































def cross(u: Union[Vector3, Expression], v: Union[Vector3, Expression]) -> Vector3:

def _to_sx(thing: object) -> ca.SX:

@overload
def scale(v: Vector3, a: symbol_expr_float) -> Vector3:
@overload
def scale(v: Point3, a: symbol_expr_float) -> Point3:
@overload
def scale(v: Expression, a: symbol_expr_float) -> Expression:

@overload
def dot(e1: TransformationMatrix, e2: Point3) -> Point3:
@overload
def dot(e1: TransformationMatrix, e2: Vector3) -> Vector3:
@overload
def dot(e1: RotationMatrix, e2: Point3) -> Point3:
@overload
def dot(e1: RotationMatrix, e2: Vector3) -> Vector3:
@overload
def dot(e1: RotationMatrix, e2: RotationMatrix) -> RotationMatrix:
@overload
def dot(e1: RotationMatrix, e2: TransformationMatrix) -> TransformationMatrix:
@overload
def dot(e1: TransformationMatrix, e2: RotationMatrix) -> TransformationMatrix:
@overload
def dot(e1: TransformationMatrix, e2: TransformationMatrix) -> TransformationMatrix:
@overload
def dot(e1: Quaternion, e2: Quaternion) -> Expression:
@overload
def dot(e1: Union[Vector3, Point3], e2: Union[Vector3, Point3]) -> Expression:
@overload
def dot(e1: Expression, e2: Expression) -> Expression:

def kron(m1: Expression, m2: Expression) -> Expression:

def trace(matrix: Union[Expression, RotationMatrix, TransformationMatrix]) -> Expression:

# def rotation_distance(a_R_b: Expression, a_R_c: Expression) -> Expression:

@overload
def vstack(list_of_matrices: List[Union[Point3, Vector3, Quaternion]]) -> Expression:
@overload
def vstack(list_of_matrices: List[TransformationMatrix]) -> Expression:
@overload
def vstack(list_of_matrices: List[Expression]) -> Expression:

@overload
def hstack(list_of_matrices: List[TransformationMatrix]) -> Expression:
@overload
def hstack(list_of_matrices: List[Expression]) -> Expression:

@overload
def diag_stack(list_of_matrices: List[TransformationMatrix]) -> Expression:
@overload
def diag_stack(list_of_matrices: List[Expression]) -> Expression:

def normalize_axis_angle(axis: Vector3, angle: symbol_expr_float) -> Tuple[Vector3, Expression]:

def axis_angle_from_rpy(roll: symbol_expr_float, pitch: symbol_expr_float, yaw: symbol_expr_float) \
        -> Tuple[Vector3, Expression]:

def cosine_distance(v0: symbol_expr_float, v1: symbol_expr_float) -> Expression:

@overload
def euclidean_distance(v1: symbol_expr_float, v2: symbol_expr_float) -> Expression:
@overload
def euclidean_distance(v1: Point3, v2: Point3) -> Expression:

def fmod(a: symbol_expr_float, b: symbol_expr_float) -> Expression:

def normalize_angle_positive(angle: symbol_expr_float) -> Expression:

def normalize_angle(angle: symbol_expr_float) -> Expression:

def shortest_angular_distance(from_angle: symbol_expr_float, to_angle: symbol_expr_float) -> Expression:

def quaternion_slerp(q1: Quaternion, q2: Quaternion, t: symbol_expr_float) -> Quaternion:

@overload
def slerp(v1: Vector3, v2: Vector3, t: symbol_expr_float) -> Vector3:
@overload
def slerp(v1: Expression, v2: Expression, t: symbol_expr_float) -> Expression:

def save_acos(angle: symbol_expr_float) -> Expression:

def entrywise_product(matrix1: Expression, matrix2: Expression) -> Expression:

def floor(x: symbol_expr_float) -> Expression:

def ceil(x: symbol_expr_float) -> Expression:

def round_up(x: symbol_expr_float, decimal_places: symbol_expr_float) -> Expression:

def round_down(x: symbol_expr_float, decimal_places: symbol_expr_float) -> Expression:

def sum(matrix: Expression) -> Expression:

def sum_row(matrix: Expression) -> Expression:

def sum_column(matrix: Expression) -> Expression:

def distance_point_to_line_segment(frame_P_current: Point3, frame_P_line_start: Point3, frame_P_line_end: Point3) \
        -> Tuple[Expression, Point3]:

def distance_point_to_line(frame_P_point: Point3, frame_P_line_point: Point3, frame_V_line_direction: Vector3) \
    -> Expression:

def distance_point_to_plane(frame_P_current: Point3, frame_V_v1: Vector3,
                                          frame_V_v2: Vector3) -> \
        Tuple[Expression, Point3]:

def distance_point_to_plane_signed(frame_P_current: Point3, frame_V_v1: Vector3,
                                          frame_V_v2: Vector3) -> \
        Tuple[Expression, Point3]:

def project_to_cone(frame_V_current: Vector3, frame_V_cone_axis: Vector3, cone_theta: Union[Symbol, float, Expression]) -> Vector3:

def project_to_plane(frame_V_plane_vector1: Vector3, frame_V_plane_vector2: Vector3, frame_P_point: Point3) -> Point3:

def angle_between_vector(v1: Vector3, v2: Vector3) -> Expression:

def velocity_limit_from_position_limit(acceleration_limit: Union[Symbol, float],
                                       position_limit: Union[Symbol, float],
                                       current_position: Union[Symbol, float],
                                       step_size: Union[Symbol, float],
                                       eps: float = 1e-5) -> Expression:

def rotational_error(r1: RotationMatrix, r2: RotationMatrix) -> Expression:

def to_str(expression: all_expressions) -> List[List[str]]:

def total_derivative(expr: Union[Symbol, Expression],
                     symbols: Iterable[Symbol],
                     symbols_dot: Iterable[Symbol]) \
        -> Expression:

def total_derivative2(expr: Union[Symbol, Expression],
                      symbols: Iterable[Symbol],
                      symbols_dot: Iterable[Symbol],
                      symbols_ddot: Iterable[Symbol]) -> Expression:

def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:

def quaternion_conjugate(q: Quaternion) -> Quaternion:

def quaternion_diff(q1: Quaternion, q2: Quaternion) -> Quaternion:

def sign(x: symbol_expr_float) -> Expression:

def cos(x: symbol_expr_float) -> Expression:

def sin(x: symbol_expr_float) -> Expression:

def exp(x: symbol_expr_float) -> Expression:

def log(x: symbol_expr_float) -> Expression:

def tan(x: symbol_expr_float) -> Expression:

def sinh(x: symbol_expr_float) -> Expression:

def cosh(x: symbol_expr_float) -> Expression:

def sqrt(x: symbol_expr_float) -> Expression:

def acos(x: symbol_expr_float) -> Expression:

def atan2(x: symbol_expr_float, y: symbol_expr_float) -> Expression:

def solve_for(expression: Expression, target_value: float, start_value: float = 0.0001, max_tries: int = 10000,
              eps: float = 1e-10, max_step: float = 1) -> float:

def one_step_change(current_acceleration: symbol_expr_float, jerk_limit: symbol_expr_float, dt: float) \
        -> Expression:

def gauss(n: symbol_expr_float) -> Expression:


def r_gauss(integral: symbol_expr_float) -> Expression:


def substitute(expression: Union[Symbol, Expression], old_symbols: List[Symbol], new_symbols: List[Union[Symbol, Expression]]) -> Expression:
   

def matrix_inverse(a: Expression) -> Expression:

def gradient(ex: Expression, arg:Expression) -> Expression:

def is_true_symbol(expr: Expression) -> bool:

def is_true3(expr: Union[Symbol, Expression]) -> Expression:
def is_true3_symbol(expr: Expression) -> Expression:

def is_false3(expr: Union[Symbol, Expression]) -> Expression:
def is_false3_symbol(expr: Expression) -> Expression:

def is_unknown3(expr: Union[Symbol, Expression]) -> Expression:
def is_unknown3_symbol(expr: Expression) -> Expression:

def is_false_symbol(expr: Expression) -> bool:

def is_constant(expr: Expression) -> bool:

def det(expr: Expression) -> Expression:

def distance_projected_on_vector(point1: Point3, point2: Point3, vector: Vector3) -> Expression:

def distance_vector_projected_on_plane(point1: Point3, point2:Point3, normal_vector: Vector3) -> Vector3:

def replace_with_three_logic(expr: Expression) -> Expression:

def is_inf(expr: Expression) -> bool:

SpatialType = TypeVar('SpatialType', Point3, TransformationMatrix, Vector3, Quaternion, RotationMatrix)